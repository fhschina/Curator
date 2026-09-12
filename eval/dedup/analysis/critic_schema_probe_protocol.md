# Actual schema-constraint capability probe

This is an eight-call synthetic transport diagnostic inside the current five-hour window, not another semantic Judge version.
The completed v3 pilot left two required-field omissions, so provider-side constraints may be more effective than more prose.

NVIDIA's [NIM structured-generation documentation](https://docs.nvidia.com/nim/large-language-models/1.15.0/structured-generation.html)
distinguishes arbitrary JSON-object mode from schema-guided output and documents `guided_json`.
Its [backend compatibility notes](https://docs.nvidia.com/nim/large-language-models/1.15.0/nim-container-variants.html)
also describe a flat `response_format` schema variant. These public NIM documents do NOT establish what our configured
`inference-api.nvidia.com` route actually supports. No backend or feature support is inferred from its model name.

Probe the existing 27B endpoint using four encodings: baseline JSON object, `guided_json`, standard nested `json_schema`,
and the documented flat schema encoding. Two trials each, no retries, 128 output tokens, temperature 0.
The prompt asks for PROMPT_VALUE while the schema (absent from prompt text) permits a different sentinel in each trial.
Following only the prompt is evidence that this request did not enforce the schema, not a successful schema test.
HTTP rejection is not a semantic Judge failure; error response bodies are not recorded to avoid credential leakage.
Passing both synthetic probes is only observed capability; the actual eight-field critic schema must still be checked.

Freeze all eight requests before sending, save raw successful responses before parsing, and keep per-attempt transport receipts.
No dataset examples, reference changes, new model, main calls, Judge scores or release claims. The original five-hour deadline applies.
