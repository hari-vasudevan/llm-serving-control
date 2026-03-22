classdef ollama_plant < matlab.System
%OLLAMA_PLANT  Real LLM inference plant via Ollama HTTP API.
%
%   Port interface:
%     Inputs  (1) lambda_k  (2) bk  (3) qk
%     Outputs (1) l_mean  (2) l_p95  (3) a_k  (4) qkp1
%
%   Two-part design to satisfy R2025b Simulink:
%   1. getSimulateUsingImpl returns Interpreted execution so stepImpl
%      runs as plain MATLAB — no codegen of webwrite/parfeval needed.
%   2. Propagation methods return all 4 port specs as multiple outputs
%      (old-style all-ports signature) so Simulink never falls back to
%      codegen to infer signal dimensions.

    properties
        ollama_url    = 'http://localhost:11434/api/generate'
        model_name    = 'qwen2.5:3b'
        num_predict   = 1
        n_win         = 20
        http_timeout  = 30
        q_max         = 12
        b_min         = 1
        b_max         = 12
        prompts_path  = '/Users/hvasudevan/Documents/MATLAB/llm_control_v2/chapter_3/llm_requirements/prompts.txt'
        n_warmup      = 4
    end

    properties (Nontunable)
        prompt_list = {'What is 2 plus 2?'}
    end

    properties (Access = private)
        lat_buf
        buf_idx
    end

    %% Public: call from setup_plant.m before simulation starts
    methods (Access = public)
        function load_prompts(obj)
            fid = fopen(obj.prompts_path, 'r');
            if fid == -1
                error('ollama_plant:io', 'Cannot open: %s', obj.prompts_path);
            end
            raw = {};
            while ~feof(fid)
                line = strtrim(fgetl(fid));
                if ischar(line) && ~isempty(line) && line(1) ~= '#'
                    cleaned = regexprep(line, '^\[\w+\]\s*', '');
                    if ~isempty(cleaned)
                        raw{end+1} = cleaned;
                    end
                end
            end
            fclose(fid);
            obj.prompt_list = raw;
            fprintf('ollama_plant: loaded %d prompts\n', numel(obj.prompt_list));
        end
    end

    methods (Access = protected)

        %% Forces Simulink to run stepImpl as plain interpreted MATLAB.
        %% No codegen of stepImpl/setupImpl is ever attempted.
        function flag = getSimulateUsingImpl(~)
            flag = 'Interpreted execution';
        end

        function num = getNumInputsImpl(~)
            num = 3;
        end

        function num = getNumOutputsImpl(~)
            num = 4;
        end

        %% All-ports propagation methods (old-style multi-output signature).
        %% Returning all 4 port specs here means Simulink NEVER needs
        %% codegen to infer signal dimensions — even in Interpreted mode.
        function [sz1,sz2,sz3,sz4] = getOutputSizeImpl(~)
            sz1 = [1 1]; sz2 = [1 1]; sz3 = [1 1]; sz4 = [1 1];
        end

        function [dt1,dt2,dt3,dt4] = getOutputDataTypeImpl(~)
            dt1 = 'double'; dt2 = 'double'; dt3 = 'double'; dt4 = 'double';
        end

        function [f1,f2,f3,f4] = isOutputFixedSizeImpl(~)
            f1 = true; f2 = true; f3 = true; f4 = true;
        end

        function [f1,f2,f3,f4] = isOutputComplexImpl(~)
            f1 = false; f2 = false; f3 = false; f4 = false;
        end

        function setupImpl(obj)
            obj.lat_buf = 500 * ones(1, obj.n_win);
            obj.buf_idx = 0;
            fprintf('ollama_plant: warming up (%d requests)...\n', obj.n_warmup);
            warm_f = cell(obj.n_warmup, 1);
            for i = 1:obj.n_warmup
                warm_f{i} = parfeval(@ollama_ttft, 1, ...
                    obj.ollama_url, obj.model_name, 'Hello', 3, obj.http_timeout);
            end
            for i = 1:obj.n_warmup
                try; fetchOutputs(warm_f{i}); catch; end
            end
            fprintf('ollama_plant: warm-up complete.\n');
        end

        function resetImpl(obj)
            % Runs when simulation resets — re-init buffer only
            obj.lat_buf = 500 * ones(1, obj.n_win);
            obj.buf_idx = 0;
        end

        function [l_mean_out, l_p95_out, a_k_out, qkp1_out] = stepImpl(obj, lambda_k, bk, qk)
            bk_safe   = max(obj.b_min, min(obj.b_max, round(double(bk))));
            n_prompts = numel(obj.prompt_list);
            a_k       = poissrnd(max(0, double(lambda_k)));
            req_f = cell(bk_safe, 1);
            for i = 1:bk_safe
                prompt = obj.prompt_list{randi(n_prompts)};
                req_f{i} = parfeval(@ollama_ttft, 1, ...
                    obj.ollama_url, obj.model_name, ...
                    prompt, obj.num_predict, obj.http_timeout);
            end
            lat_samples = zeros(1, bk_safe);
            for i = 1:bk_safe
                try
                    lat_samples(i) = fetchOutputs(req_f{i});
                catch
                    lat_samples(i) = obj.http_timeout * 1000;
                end
            end
            for i = 1:bk_safe
                obj.buf_idx = mod(obj.buf_idx, obj.n_win) + 1;
                obj.lat_buf(obj.buf_idx) = lat_samples(i);
            end
            data       = obj.lat_buf;
            l_mean_out = mean(data);
            sorted     = sort(data);
            idx_95     = max(1, ceil(0.95 * numel(sorted)));
            l_p95_out  = sorted(idx_95);
            qkp1_out   = max(0, min(obj.q_max, double(qk) + double(a_k) - bk_safe));
            a_k_out    = double(a_k);
        end

    end

    methods (Static, Access = public)

        % send_request moved to standalone ollama_ttft.m so parfeval
        % workers can find it by name without class-path resolution.
        function lat_ms = send_request(url, mdl, prompt, n_pred, timeout_sec)
            % Thin wrapper kept for backward compatibility
            lat_ms = ollama_ttft(url, mdl, prompt, n_pred, timeout_sec);
            %SEND_REQUEST  Measure Time To First Token (TTFT) via streaming API.
            %
            % Why TTFT and not end-to-end:
            %   End-to-end = TTFT + n_tokens * ~52ms/token.
            %   The generation component (n_tokens * 52ms) is driven by
            %   response length, not server load — it is noise for the
            %   queue-depth controller.  TTFT reflects only prefill cost,
            %   which is directly controlled by batch size and queue depth.
            %
            % Measured on M2 qwen2.5:7b:
            %   TTFT:          ~130-195 ms (what we want to control)
            %   Generation:    ~52 ms/token (noise — excluded here)
            %   End-to-end:    ~600 ms for 9 tokens
            %
            % Implementation: stream=true, read first chunk, then close.
            % Uses Java HttpURLConnection — the only way to get true
            % streaming in MATLAB without toolboxes.
            % n_pred is kept as a parameter for API compatibility but
            % we stop reading after the first token arrives.

            timeout_ms = int32(timeout_sec * 1000);

            % Build streaming JSON body
            body = sprintf(['{"model":"%s","prompt":"%s",', ...
                '"stream":true,', ...
                '"options":{"num_predict":%d}}'], ...
                mdl, prompt, n_pred);

            % Open HTTP connection
            url_obj = java.net.URL(url);
            conn    = url_obj.openConnection();
            conn.setRequestMethod('POST');
            conn.setRequestProperty('Content-Type', 'application/json');
            conn.setDoOutput(true);
            conn.setConnectTimeout(timeout_ms);
            conn.setReadTimeout(timeout_ms);

            % Send request body
            out = conn.getOutputStream();
            out.write(int8(body));
            out.flush();
            out.close();

            % Start timer AFTER request is sent — measures server-side
            % queue + prefill time only, not network serialisation.
            t_start = tic;

            % Read first chunk — this unblocks as soon as the first
            % token is generated (i.e. at TTFT).
            reader = java.io.BufferedReader(java.io.InputStreamReader(conn.getInputStream()));
            reader.readLine();   % blocks until first token arrives
            lat_ms = toc(t_start) * 1000;

            % Close immediately — we do not need the rest of the response.
            reader.close();
            conn.disconnect();
        end

    end

end
