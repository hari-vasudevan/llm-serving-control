classdef ollama_plant < matlab.System
%OLLAMA_PLANT  Real LLM inference plant via Ollama HTTP API.
%
%   Replaces the stochastic MATLAB Function plant (llm_plant.m) in Simulink.
%   Drop a MATLAB System block in place of the fcn block inside Plant1,
%   set the system object name to 'ollama_plant', and wire ports identically.
%
%   Port interface (identical to the stochastic fcn block):
%     Inputs  (1) lambda_k  -- mean arrival rate this tick  [req/tick]
%             (2) bk        -- batch size (concurrency)     [requests]
%             (3) qk        -- current queue depth          [requests]
%
%     Outputs (1) l_mean    -- mean latency this tick       [ms]
%             (2) l_p95     -- rolling empirical p95        [ms]
%             (3) a_k       -- actual Poisson arrivals      [requests]
%             (4) qkp1      -- queue depth at next tick     [requests]
%
%   How it works each tick:
%     1. Draw true Poisson arrivals:  a_k ~ Poisson(lambda_k)
%     2. Fire bk concurrent HTTP requests to Ollama (via parfeval)
%     3. Collect wall-clock latencies as responses arrive
%     4. Update circular buffer; compute empirical mean and p95
%     5. Update queue:  qkp1 = max(0, qk + a_k - completions)
%
%   Concurrency model:
%     bk requests are sent in parallel using parfeval on the parallel pool.
%     Ollama processes up to OLLAMA_NUM_PARALLEL simultaneously (set to 4
%     via launchd). Requests beyond that queue inside Ollama, incurring
%     real waiting latency — giving the cascade controller real dynamics.
%
%   Naming convention: all lowercase, underscore-separated (project standard).

    % ------------------------------------------------------------------
    % Public tunable properties (readable from setup_plant.m struct)
    % ------------------------------------------------------------------
    properties
        % Ollama server
        ollama_url     = 'http://localhost:11434/api/generate'
        model_name     = 'qwen2.5:7b'

        % Max tokens per response — controls response length and
        % therefore inference time. Shorter = lower latency floor.
        num_predict    = 50

        % Rolling window length for empirical p95 [samples]
        n_win          = 20

        % Request timeout [s] — abort a request if it takes longer.
        % Set well above expected max latency to avoid false timeouts.
        http_timeout   = 30

        % Queue upper bound (for clamping qkp1)
        q_max          = 3000

        % Batch size bounds
        b_min          = 1
        b_max          = 64

        % Full path to prompts.txt
        prompts_path   = '/Users/hvasudevan/Documents/MATLAB/llm_control_v2/chapter_3/llm_requirements/prompts.txt'

        % Warm-up: fire this many dummy requests in setupImpl to load
        % model weights into GPU memory before simulation starts.
        n_warmup       = 4
    end

    % ------------------------------------------------------------------
    % Private state
    % ------------------------------------------------------------------
    properties (Access = private)
        % Circular latency buffer for rolling p95
        lat_buf        % 1 x n_win double
        buf_idx        % current write position (1-indexed)
        buf_full       % logical: has buffer completed first pass?

        % Loaded prompt list (cell array of strings)
        prompt_list

        % Parallel pool handle (created once in setupImpl)
        par_pool
    end

    % ------------------------------------------------------------------
    % Setup — runs once when simulation starts
    % ------------------------------------------------------------------
    methods (Access = protected)

        function setup_impl(obj)
            % -- 1. Load prompt library ------------------------------------
            % Read prompts.txt, skip comment lines (starting with #),
            % strip category tags [SHORT]/[MEDIUM]/[LONG].
            fid = fopen(obj.prompts_path, 'r');
            if fid == -1
                error('ollama_plant: cannot open prompts file: %s', obj.prompts_path);
            end
            raw_lines = {};
            while ~feof(fid)
                line = strtrim(fgetl(fid));
                if ischar(line) && ~isempty(line) && line(1) ~= '#'
                    % Strip category tag: "[SHORT] What is..." -> "What is..."
                    cleaned = regexprep(line, '^\[\w+\]\s*', '');
                    if ~isempty(cleaned)
                        raw_lines{end+1} = cleaned; %#ok<AGROW>
                    end
                end
            end
            fclose(fid);
            obj.prompt_list = raw_lines;
            fprintf('ollama_plant: loaded %d prompts from %s\n', ...
                    numel(obj.prompt_list), obj.prompts_path);

            % -- 2. Initialise rolling latency buffer ----------------------
            % Warm-start at a reasonable baseline (500 ms) so the outer
            % loop sees near-zero error at t=0 rather than a cold buffer.
            obj.lat_buf  = 500 * ones(1, obj.n_win);
            obj.buf_idx  = 0;
            obj.buf_full = true;

            % -- 3. Open parallel pool ------------------------------------
            % Reuse existing pool if already open (avoids 10 s startup
            % penalty on every simulation run).
            obj.par_pool = gcp('nocreate');
            if isempty(obj.par_pool)
                fprintf('ollama_plant: starting parallel pool...\n');
                obj.par_pool = parpool('local', obj.b_max);
            end

            % -- 4. Warm-up requests --------------------------------------
            % Fire n_warmup requests to pull model weights into GPU memory.
            % Warm-up latency is real but not recorded in the rolling buffer.
            fprintf('ollama_plant: warming up model (%d requests)...\n', obj.n_warmup);
            warmup_futures = obj.fire_requests(obj.n_warmup);
            for i = 1:obj.n_warmup
                try
                    fetchOutputs(warmup_futures(i));
                catch
                    % ignore warmup errors silently
                end
            end
            fprintf('ollama_plant: warm-up complete. Ready.\n');
        end

        % ------------------------------------------------------------------
        % Step — runs every simulation tick
        % ------------------------------------------------------------------
        function [l_mean_out, l_p95_out, a_k_out, qkp1_out] = step_impl(obj, lambda_k, bk, qk)
            % Clamp bk to physical limits
            bk_safe = max(obj.b_min, min(obj.b_max, round(bk)));

            % -- 1. Poisson arrivals ----------------------------------------
            a_k = poissrnd(max(0, lambda_k));

            % -- 2. Fire bk concurrent requests ----------------------------
            futures = obj.fire_requests(bk_safe);

            % -- 3. Collect latencies  -------------------------------------
            % Wait for all futures; record wall-clock latency of each.
            % Timed-out or errored requests get a penalty latency.
            lat_samples = zeros(1, bk_safe);
            for i = 1:bk_safe
                try
                    result = fetchOutputs(futures(i));
                    lat_samples(i) = result;
                catch
                    % Request failed or timed out — assign penalty latency
                    lat_samples(i) = obj.http_timeout * 1000;
                end
            end

            % -- 4. Count completions and update rolling buffer -------------
            n_completed = bk_safe;   % all requests either complete or timeout
            for i = 1:n_completed
                obj.buf_idx = mod(obj.buf_idx, obj.n_win) + 1;
                obj.lat_buf(obj.buf_idx) = lat_samples(i);
            end
            obj.buf_full = true;   % always true after setupImpl warm-start

            % -- 5. Rolling statistics  ------------------------------------
            data       = obj.lat_buf;
            l_mean_out = mean(data);
            sorted     = sort(data);
            idx_95     = max(1, ceil(0.95 * numel(sorted)));
            l_p95_out  = sorted(idx_95);

            % -- 6. Queue update  ------------------------------------------
            qkp1_out = max(0, min(obj.q_max, qk + a_k - n_completed));
            a_k_out  = double(a_k);
        end

        % ------------------------------------------------------------------
        % Port size/type declarations (required by matlab.System)
        % ------------------------------------------------------------------
        function num = get_num_inputs_impl(~)
            num = 3;   % lambda_k, bk, qk
        end

        function num = get_num_outputs_impl(~)
            num = 4;   % l_mean, l_p95, a_k, qkp1
        end

        function flag = is_output_fixed_size_impl(~, ~)
            flag = true;
        end

        function sz = get_output_size_impl(~, ~)
            sz = [1 1];
        end

        function dt = get_output_data_type_impl(~, ~)
            dt = 'double';
        end

        function flag = is_output_complex_impl(~, ~)
            flag = false;
        end

        function [sz1, sz2, sz3] = get_input_size_impl(~, ~, ~, ~)
            sz1 = [1 1]; sz2 = [1 1]; sz3 = [1 1];
        end

    end   % protected methods

    % ------------------------------------------------------------------
    % Private helpers
    % ------------------------------------------------------------------
    methods (Access = private)

        function futures = fire_requests(obj, n)
            %FIRE_REQUESTS  Launch n concurrent Ollama HTTP calls via parfeval.
            %
            % Returns a 1×n array of parallel.Future objects.
            % Each future resolves to a scalar double: wall-clock latency [ms].

            n_prompts = numel(obj.prompt_list);
            futures(n) = parallel.Future;   % pre-allocate

            for i = 1:n
                % Uniform random prompt selection
                idx    = randi(n_prompts);
                prompt = obj.prompt_list{idx};

                % Capture variables needed inside parfeval closure
                url         = obj.ollama_url;
                mdl         = obj.model_name;
                n_pred      = obj.num_predict;
                timeout_sec = obj.http_timeout;

                futures(i) = parfeval(@ollama_plant.send_request, 1, ...
                    url, mdl, prompt, n_pred, timeout_sec);
            end
        end

    end   % private methods

    % ------------------------------------------------------------------
    % Static helpers (must be static to be callable inside parfeval)
    % ------------------------------------------------------------------
    methods (Static, Access = private)

        function lat_ms = send_request(url, mdl, prompt, n_pred, timeout_sec)
            %SEND_REQUEST  Single blocking Ollama HTTP call.
            %
            % Sends one generate request, returns wall-clock latency [ms].
            % This function runs on a parallel worker — no object state access.

            payload = struct(...
                'model',   mdl, ...
                'prompt',  prompt, ...
                'stream',  false, ...
                'options', struct('num_predict', n_pred));

            opts = weboptions(...
                'MediaType',     'application/json', ...
                'Timeout',       timeout_sec, ...
                'RequestMethod', 'post');

            t_start = tic;
            try
                webwrite(url, payload, opts);
            catch
                % timeout or connection error — caller handles penalty
                rethrow(lasterror); %#ok<LERR>
            end
            lat_ms = toc(t_start) * 1000;
        end

    end   % static methods

end   % classdef
