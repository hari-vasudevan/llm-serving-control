function lat_ms = vllm_e2e(gen_url, model, prompt, max_tokens, timeout_sec)
%VLLM_E2E  Measure end-to-end latency from a vLLM server.
%
%   Uses the non-streaming /v1/completions endpoint (stream=false).
%   Wall-clock time from request sent to full response received.
%
%   Why end-to-end and not TTFT for the cascade controller:
%     TTFT closes the scheduler slot after the first token, so requests
%     complete in ~50-100ms even for a 0.6B model.  vLLM drains the
%     scheduler queue between ticks and num_requests_waiting stays at 0.
%
%     End-to-end keeps the scheduler slot occupied for max_tokens steps
%     (~20-50ms/token on a 0.6B model).  With max_tokens=20 and
%     max_num_seqs=4, requests in excess of 4 genuinely queue in vLLM's
%     scheduler, and num_requests_waiting > 0 is observable and real.
%
%     End-to-end latency is also the correct metric for production
%     SLA compliance -- it is what the end user experiences.
%
%   This function is standalone so parfeval workers can find it by name.

opts    = weboptions( ...
    'MediaType',     'application/json', ...
    'Timeout',       timeout_sec, ...
    'RequestMethod', 'post');

prompt_safe = strrep(prompt, '"', "'");
payload = struct( ...
    'model',      model, ...
    'prompt',     prompt_safe, ...
    'max_tokens', max_tokens, ...
    'stream',     false);

t_start = tic;
webwrite(gen_url, payload, opts);
lat_ms  = toc(t_start) * 1000;

end
