# Selected-v3 transport-only paired comparison

Use the original 19 pilot rows, all original payloads, fixed .12 main, unchanged references and two repeats.
Three fresh arms: original .12 critic; v3 with JSON-object mode; the exact same v3 messages with nested strict schema.
Only omit uniqueItems from server schema after all three actual compatibility probes validate locally.
The local v3 validator still rejects duplicates, missing/foreign IDs, empty contexts and misaligned spans.
Keep eight required fields and all type/enum/length/additional-property restrictions otherwise unchanged.
No semantic prompt edits, no schema retries, no response repair. Preserve every raw response and every denominator.
This isolates transport validity from semantic judgment. H0850 remains a known semantic negative protection.
Review every mismatch before any further semantic fix. Earlier pilot expansion gates and the five-hour deadline remain.
