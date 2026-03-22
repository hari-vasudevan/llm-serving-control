classdef vllm_plant < matlab.System
%VLLM_PLANT  Real vLLM inference plant for Chapter 5 Simulink.
%
% Replaces the statistical MATLAB Function block inside Plant1.
%
% KEY DIFFERENCES FROM ollama_plant (Chapter 3/4):
%
%   1. q[k] is read from vLLM's /metrics endpoint at the START of each
%      tick -- it is a directly observed scheduler state, not a software
%      counter updated via a z^-1 delay.  The Delay block for q inside
%      Plant1 is therefore REMOVED.
%
%   2. Latency is end-to-end (e2e), not TTFT.  With max_num_seqs=4 and
%      B[k] up to 8, requests take ~250-500ms each.  At dt=1s, B<=4 fits
%      inside one tick; B>4 may spill but the blocking fetchOutputs
%      naturally clamps the tick to the actual completion time.
%
%   3. Arrivals: we simulate Poisson arrivals (as in the stochastic plant)
%      rather than driving them from outside.  The controller sets B[k];
%      we fire B[k] concurrent requests to vLLM and let vLLM queue the
%      excess.
%
% SIMULINK PORTS
%   Inputs:  (1) lambda_k  -- arrival rate [req/tick]
%            (2) bk        -- batch size commanded by controller
%   Outputs: (1) l_mean    -- rolling mean e2e latency [ms]
%            (2) l_p95     -- rolling p95 e2e latency [ms]
%            (3) qk        -- num_requests_waiting from /metrics
%            (4) ak        -- Poisson arrivals this tick
%
% SIMULINK DIALOG PARAMETERS
%   ollama_url, model_name, e2e_tokens, n_win, b_min, b_max,
%   q_max, n_warmup, http_timeout, prompts_path
%
% BLOCK SETTINGS (required):
%   "Simulate using" -> "Interpreted execution"
%   Sample time: perturbed.dt  (= 1.0 s)

    % -----------------------------------------------------------------
    % Tunable parameters (visible in block dialog)
    % -----------------------------------------------------------------
    properties (Nontunable)
        vllm_url     = 'http://localhost:8001/v1/completions'
        metrics_url  = 'http://localhost:8001/metrics'
        model_name   = 'mlx-community/Qwen3-0.6B-4bit'
        e2e_tokens   = 20        % output tokens per request
        n_win        = 20        % rolling buffer length
        b_min        = 1
        b_max        = 8
        q_max        = 4         % max_num_seqs
        n_warmup     = 3
        http_timeout = 30        % s
        prompts_path = ''        % path to prompts.txt
    end

    properties (Access = private)
        lat_buf      % rolling latency buffer  [1 x n_win]
        buf_idx      % current write index
        prompt_list  % cell array of prompts
    end

    % -----------------------------------------------------------------
    % System methods
    % -----------------------------------------------------------------
    methods (Access = protected)

        function setupImpl(obj)
            % Warm-start latency buffer at nominal value
            obj.lat_buf = 800 * ones(1, obj.n_win);
            obj.buf_idx = 0;

            % Load prompts
            if ~isempty(obj.prompts_path) && exist(obj.prompts_path, 'file')
                fid  = fopen(obj.prompts_path, 'r');
                raw  = textscan(fid, '%s', 'Delimiter', '\n', 'Whitespace', '');
                fclose(fid);
                obj.prompt_list = raw{1};
                obj.prompt_list = obj.prompt_list(~cellfun('isempty', obj.prompt_list));
            else
                obj.prompt_list = {'What is 2+2?'; 'Name a colour.'; 'Capital of France?'};
            end
            fprintf('[vllm_plant] Loaded %d prompts.\n', numel(obj.prompt_list));

            % Warm up
            fprintf('[vllm_plant] Warming up (%d requests)...\n', obj.n_warmup);
            pool = gcp('nocreate');
            if ~isempty(pool)
                addAttachedFiles(pool, fileparts(mfilename('fullpath')));
            end
            for i = 1:obj.n_warmup
                try
                    f = parfeval(@vllm_e2e, 1, obj.vllm_url, obj.model_name, ...
                        'Hello', obj.e2e_tokens, obj.http_timeout);
                    lat = fetchOutputs(f);
                    fprintf('  warmup %d: %.0f ms\n', i, lat);
                catch e
                    fprintf('  warmup %d: error (%s)\n', i, e.message);
                end
            end
            fprintf('[vllm_plant] Ready.\n');
        end

        function [l_mean, l_p95, q_k, a_k] = stepImpl(obj, lambda_k, bk)
            % ----------------------------------------------------------------
            % 1. Read q[k] from /metrics at START of tick.
            %    This is the queue left from previous tick's requests.
            % ----------------------------------------------------------------
            q_k = 0;
            try
                raw = webread(obj.metrics_url, ...
                    weboptions('Timeout', 3, 'ContentType', 'text'));
                q_k = obj.parse_waiting(raw);
            catch
                % metrics read failed -- use 0
            end
            q_k = min(q_k, obj.q_max);

            % ----------------------------------------------------------------
            % 2. Simulate Poisson arrivals
            % ----------------------------------------------------------------
            a_k = poissrnd(max(0, lambda_k));

            % ----------------------------------------------------------------
            % 3. Clamp batch size
            % ----------------------------------------------------------------
            b_act = min(max(round(bk), obj.b_min), obj.b_max);

            % ----------------------------------------------------------------
            % 4. Fire b_act concurrent requests and collect latencies
            % ----------------------------------------------------------------
            futures = cell(b_act, 1);
            for i = 1:b_act
                prompt = obj.prompt_list{mod(i-1, numel(obj.prompt_list))+1};
                futures{i} = parfeval(@vllm_e2e, 1, obj.vllm_url, ...
                    obj.model_name, prompt, obj.e2e_tokens, obj.http_timeout);
            end

            lats = zeros(1, b_act);
            for i = 1:b_act
                try
                    lats(i) = fetchOutputs(futures{i});
                catch
                    lats(i) = NaN;
                end
            end

            % ----------------------------------------------------------------
            % 5. Update rolling buffer
            % ----------------------------------------------------------------
            for i = 1:b_act
                if ~isnan(lats(i))
                    obj.buf_idx = mod(obj.buf_idx, obj.n_win) + 1;
                    obj.lat_buf(obj.buf_idx) = lats(i);
                end
            end

            % ----------------------------------------------------------------
            % 6. Compute rolling statistics
            % ----------------------------------------------------------------
            l_mean = mean(obj.lat_buf);
            sorted = sort(obj.lat_buf);
            idx95  = max(1, ceil(0.95 * obj.n_win));
            l_p95  = sorted(idx95);
        end

        % -----------------------------------------------------------------
        % Port size / type declarations (required for Simulink)
        % -----------------------------------------------------------------
        function [s1, s2, s3, s4] = getOutputSizeImpl(~, ~, ~)
            s1 = [1 1]; s2 = [1 1]; s3 = [1 1]; s4 = [1 1];
        end

        function [d1, d2, d3, d4] = getOutputDataTypeImpl(~, ~, ~)
            d1 = 'double'; d2 = 'double'; d3 = 'double'; d4 = 'double';
        end

        function [c1, c2, c3, c4] = isOutputComplexImpl(~, ~, ~)
            c1 = false; c2 = false; c3 = false; c4 = false;
        end

        function [f1, f2, f3, f4] = isOutputFixedSizeImpl(~, ~, ~)
            f1 = true; f2 = true; f3 = true; f4 = true;
        end

        function flag = getSimulateUsingImpl(~)
            flag = 'Interpreted execution';
        end

    end

    % -----------------------------------------------------------------
    % Private helpers
    % -----------------------------------------------------------------
    methods (Access = private)
        function q = parse_waiting(~, raw_text)
            q = 0;
            lines = strsplit(raw_text, newline);
            for i = 1:numel(lines)
                line = strtrim(lines{i});
                if isempty(line) || line(1) == '#'; continue; end
                clean = strtrim(regexprep(line, '\{[^}]*\}', ''));
                parts = regexp(clean, '\s+', 'split');
                if numel(parts) < 2; continue; end
                if contains(parts{1}, 'num_requests_waiting')
                    v = str2double(parts{2});
                    if ~isnan(v); q = q + v; end
                end
            end
        end
    end

end
