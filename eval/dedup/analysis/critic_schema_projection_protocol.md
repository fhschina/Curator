# Critic schema transport compatibility

The nested schema sentinel worked twice; all three complete critic schemas received actual upstream 400s.
Freeze six compatibility requests: the identical H0701/H0850/H0893 v3 messages and generation parameters,
each with (1) only uniqueItems removed or (2) uniqueItems and string length keywords removed.
Keep all eight required fields, enums, types and additionalProperties:false. No local validation is removed.
Run each request through a separate relay so an expected schema rejection cannot contaminate subsequent probes.
No retries, no semantic score, no reference edits, no passing benchmark claim. Preserve raw before validation.
Prefer the least projected schema only if all three actual responses pass the unchanged local v3 contract.
This is finite compatibility evidence, not a guarantee of semantic or schema accuracy in a larger population.
