function lat_ms = vllm_ttft(gen_url, model, prompt, max_tokens, timeout_sec)
%VLLM_TTFT  Measure Time To First Token from a vLLM server via streaming.
%
%   vLLM uses the OpenAI-compatible /v1/completions endpoint with
%   stream=true.  Each SSE chunk is prefixed with "data: ".
%   The first non-empty chunk arriving = first token = TTFT.
%
%   Inputs
%     gen_url     full URL to /v1/completions endpoint
%     model       model name as registered with vLLM
%     prompt      prompt string
%     max_tokens  max tokens to generate (1 = minimise generation component)
%     timeout_sec HTTP connect + read timeout in seconds
%
%   Output
%     lat_ms      TTFT in milliseconds (wall-clock from request sent to
%                 first token received)
%
%   This function is standalone so parfeval workers can find it by name
%   without needing a class or any workspace state.

timeout_ms = int32(timeout_sec * 1000);

% Build JSON body for OpenAI-compatible streaming completion
% prompt is escaped minimally -- strip double quotes to be safe
prompt_safe = strrep(prompt, '"', "'");
body = sprintf(['{"model":"%s","prompt":"%s",' ...
    '"max_tokens":%d,"stream":true}'], ...
    model, prompt_safe, max_tokens);

% Open HTTP connection via Java (only way to get true streaming in MATLAB)
url_obj = java.net.URL(gen_url);
conn    = url_obj.openConnection();
conn.setRequestMethod('POST');
conn.setRequestProperty('Content-Type',  'application/json');
conn.setRequestProperty('Accept',        'text/event-stream');
conn.setDoOutput(true);
conn.setConnectTimeout(timeout_ms);
conn.setReadTimeout(timeout_ms);

% Send request
out = conn.getOutputStream();
out.write(int8(body));
out.flush();
out.close();

% Start timer AFTER request is sent
t_start = tic;

% Read SSE stream line by line.
% First line starting with "data: " and containing "text" = first token.
reader = java.io.BufferedReader(java.io.InputStreamReader(conn.getInputStream()));
while true
    line = reader.readLine();
    if isempty(line) || line.length() == 0
        continue;
    end
    line_str = char(line);
    % SSE data lines start with "data: "
    if startsWith(line_str, 'data: ')
        payload = strtrim(line_str(7:end));
        if strcmp(payload, '[DONE]')
            % Stream ended without a token -- unusual but handle gracefully
            lat_ms = toc(t_start) * 1000;
            break;
        end
        % Any non-[DONE] data line = first token arrived
        lat_ms = toc(t_start) * 1000;
        break;
    end
end

reader.close();
conn.disconnect();
