function lat_ms = ollama_ttft(url, mdl, prompt, n_pred, timeout_sec)
%OLLAMA_TTFT  Measure Time To First Token (TTFT) for one Ollama request.
%
%   Runs as a standalone function so parfeval workers can find it by
%   name without needing the ollama_plant class on their path.
%
%   Inputs
%     url          Ollama generate endpoint (string)
%     mdl          model name e.g. 'qwen2.5:7b'
%     prompt       prompt string
%     n_pred       num_predict (kept for API consistency; we stop at token 1)
%     timeout_sec  HTTP connect + read timeout [s]
%
%   Output
%     lat_ms       TTFT in milliseconds
%
%   Why TTFT and not end-to-end:
%     end-to-end = TTFT + n_tokens * ~52 ms/token
%     The generation term is driven by response length, not server load.
%     TTFT reflects only prefill cost — directly controlled by batch size
%     and queue depth — so it is the right signal for the cascade controller.
%
%   Implementation: stream=true, open Java HttpURLConnection, read the
%   first newline-delimited JSON chunk, return immediately.

timeout_ms = int32(timeout_sec * 1000);

% Build streaming JSON body
body = sprintf('{"model":"%s","prompt":"%s","stream":true,"options":{"num_predict":%d}}', ...
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

% Timer starts AFTER request is sent — measures server-side
% queue wait + prefill time only, not local serialisation cost.
t_start = tic;

% readLine() blocks until the first token chunk arrives (= TTFT)
reader = java.io.BufferedReader( ...
    java.io.InputStreamReader(conn.getInputStream()));
reader.readLine();
lat_ms = toc(t_start) * 1000;

% Close immediately — rest of the response is not needed
reader.close();
conn.disconnect();

end
