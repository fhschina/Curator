# Isolated schema confirmation

The first capability test made four actual upstream requests, not eight: its unsupported flat-schema request
opened the relay circuit and four subsequent logical requests were rejected locally. Preserve all original receipts.
One nested-standard-schema sentinel succeeded; guided_json did not enforce the sentinel. Neither observation is
a complete compatibility guarantee for our actual critic contract.

Use six independently isolated, globally paced requests with no retries: one JSON-object control, two distinct
nested-schema sentinels, then the complete unmodified v3 response schema on H0701, H0893 and H0850.
For the three real cases, retain exactly the v3 prompt, model, temperature, payload and output budget;
only change response_format from json_object to the observed standard nested schema encoding.
No schema projection, dropped keywords, altered labels or fixed outputs. Preserve every raw response before validation.
The five-hour deadline and existing key apply. This is compatibility diagnosis, not a scored new model version.
If successful, a subsequent schema-mode paired experiment is still required; identity correctness is not implied.
