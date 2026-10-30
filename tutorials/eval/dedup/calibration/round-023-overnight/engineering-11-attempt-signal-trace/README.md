# Main-attempt direction signals

This separate diagnostic reads every available main attempt, including failed
retries. It extracts explicit direction signals from native lexical JSON and,
for diagnostic signals only, complete bare JSON. It never validates a reply,
changes a decision, or grants agreement credit. Fixed references and scores
stay unchanged. Whole-pair alternatives use the native maximum-match rule.
Unlabeled diagnostics receive no correctness comparison; unknown values stay
unknown. Counts are attempt signals, including repeats, rather than accuracy.

The original complete traces and HTTP bodies remain external. The checked-in
projections contain only recognized direction signals, native attempt status,
frozen-reference comparisons and receipt hashes. They contain no raw response
body or source text. Two actual failed-attempt checks confirm that otherwise
matching raw signals still leave their native rows invalid and uncredited.
