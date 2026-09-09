# V0.6.2.4 preflight review revision

Inspection of the eight remaining V0.6.2.3 under-group rows identified three additional old-policy label conflicts. `H0807`
adds preference-based advertising to a statistical-cookie notice, `H0802` compares materially different cookie/GDPR and
ad-blocker messages, and `H0921` adds a substantive multi-sentence forum description to a registration notice.

`H0921` revises the earlier 85-row adjudication from `YES/YES` to `NO/NO`. The original decision remains in
`v0622_priority_review_adjudications.csv`; the revision is recorded separately in `v0624_preflight_adjudications.csv`. This
preserves the audit trail instead of rewriting the earlier review artifact.

After this revision, the V0.6.2.3 residual smoke contains 46 over-group and five under-group errors. The five retained
under-group labels are `H0038`, `H0347`, `H0453`, `H0679`, and `H0748`: repeated placeholder text, cookie notices with only
learn-more/accept/wishlist chrome, a legal notice with a single navigation label, and a membership pitch with a byline/header.
These define the benign-surface regression set for the next prompt.

