addpath('/Users/hvasudevan/Documents/MATLAB/llm_control_v2/chapter_5/src');
url   = 'http://localhost:8001/v1/completions';
model = 'mlx-community/Qwen3-0.6B-4bit';

t0 = tic;
for i = 1:8
    parfeval(@vllm_e2e, 1, url, model, 'What is 2+2?', 20, 30);
end
fprintf('Submitted in %.0fms. Pausing to 1s...\n', toc(t0)*1000);
elapsed = toc(t0);
if elapsed < 1.0; pause(1.0 - elapsed); end

raw = webread('http://localhost:8001/metrics', weboptions('Timeout',3,'ContentType','text'));
running = 0; waiting = 0;
for line = strsplit(raw, newline)
    s = strtrim(line{1});
    if isempty(s)||s(1)=='#'; continue; end
    c = strtrim(regexprep(s,'\{[^}]*\}',''));
    p = regexp(c,'\s+','split');
    if numel(p)<2; continue; end
    v = str2double(p{2});
    if isnan(v); continue; end
    if contains(p{1},'num_requests_waiting'); waiting = waiting + v; end
    if contains(p{1},'num_requests_running');  running = running + v; end
end
fprintf('At t=%.0fms: running=%g  waiting=%g\n', toc(t0)*1000, running, waiting);
