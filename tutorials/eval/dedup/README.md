# Fuzzy dedup evaluation example

Fuzzy deduplication groups near-duplicate documents and picks one "keeper" per group, removing the rest. This example checks whether those decisions were actually correct, by having an LLM judge look at the pairs of documents fuzzy dedup grouped together (and some it didn't) and render its own verdict on whether they're really duplicates. It doesn't change fuzzy dedup itself -- it's a way to audit and spot-check the decisions a completed fuzzy dedup run already made.

The main evaluation workflow has been tested at the 10 million and 100 million document scale, and the judge config in `judge_config/` was tuned on 8xH100 nodes. The optional coverage step below has only been validated on small inputs. The LLM judge is an automated opinion, not ground truth: treat disagreements between the judge and fuzzy dedup as things to look into, not proven errors.

## Pipeline

The example is seven standalone scripts (step 6 is optional), run in order. Each also has a full example invocation and flag reference in its own docstring.

**1. `1_data_prep.py`** downloads and extracts a Common Crawl sample with jusText. This just gets you a realistic corpus to run fuzzy dedup against -- Common Crawl pages are full of real near-duplicate content (cookie notices, legal boilerplate, syndicated articles), which is exactly the kind of thing this eval is meant to check.

**2. `2_run_fuzzy_dedup.py`** runs `FuzzyDeduplicationWorkflow` in identification-only mode: it groups near-duplicate documents and decides which one to keep in each group, but doesn't actually delete anything. Its output -- duplicate groups and removal decisions -- is what the rest of the pipeline evaluates.

**3. `3_build_pair_dataset.py`** turns fuzzy dedup's output into a dataset of document *pairs*, since that's the unit the LLM judge reasons about ("are these two documents duplicates?") rather than whole groups. Pick one of three `--pair-strategy` values depending on what you want to check:

- `keeper_removed` -- pairs the document fuzzy dedup kept with each document it removed from that group. This is the core check: "was this removal decision correct?"
- `all_pairwise` -- pairs every document in a group with every other document in that group, including removed-vs-removed pairs `keeper_removed` skips. More expensive, but also catches cases where fuzzy dedup grouped two documents together that shouldn't have been in the same group at all.
- `cross_group_sample` -- samples pairs of documents from *different* groups, i.e. pairs fuzzy dedup did **not** treat as duplicates. This checks the opposite failure mode: documents that should have been grouped together but weren't (missed duplicates).

Run the script once per strategy you want to evaluate, each with its own `--output-path`. Every pair is labeled `expected_duplicate=true` (for `keeper_removed`/`all_pairwise`) or `false` (for `cross_group_sample`) -- this is fuzzy dedup's original decision, and step 7 later checks the judge's verdict against it.

**4. `4_span_alignment.py`** is where the pairs get prepped for judging. The judge is only shown a limited amount of text per document (6000 characters, matching the judge's context window), and rather than hand it two long, mostly-identical blobs of text and hope it spots the difference, this step runs a text diff (`difflib`) between each pair and breaks the visible text into labeled spans: `SHARED` (text common to both), `A_ONLY` (only in document A), and `B_ONLY` (only in document B). That span breakdown -- not the raw text -- is what actually gets rendered into the judge's prompt, so the judge's job becomes "look at what's different" instead of "read two documents and find what's different." If a document had to be cut off to fit the 6000-character limit, the pair is flagged `truncated`, since a difference could be hiding in the part the judge never saw.

**5. `5_run_llm_judge.py`** is the actual evaluation step, and the main point of this example: it runs `LLMJudgeWorkflow` to have a locally-served LLM judge each pair against the rubric in `judge_config/fuzzy_pair_judge.yaml`. For each pair, the judge scores several rubric fields (e.g. `relation_type` -- is this pair `exact`, `near_surface`, `containment`, `version_related`, `unrelated`, etc. -- plus supporting fields like `material_difference` and `confidence_tier`) and returns its reasoning for each. `relation_type` is the main verdict step 7 evaluates. Before running this step, edit `judge_config/fuzzy_pair_judge.yaml`: set `models[0].model` to a local model path or HF repo id, and size `num_replicas`/`tensor_parallel_size` to the GPUs you have available. `LLMJudgeWorkflow` only supports locally-served models -- there's no hosted-inference-API backend.

**6. `6_run_critics.py`** optionally reviews saved main results using the coverage critic. Add `--subject` to follow coverage with subject-binding proposals and fixed-proof verification. It reuses the main YAML's model/server settings and source-stage worker settings, and runs prepare → Data Designer with native `SkipConfig` → apply inside one Pipeline. It starts its own inference service; steps 5 and 6 do not share a running server. It does not rerun main judges or their score filters, so coverage can be rerun against fixed main results.

**7. `7_analyze_results.py`** is where you actually find out how fuzzy dedup did. It takes the judge's `relation_type` verdicts and buckets them into duplicate / not_duplicate / unresolved, then compares that against each pair's `expected_duplicate` label from step 3 to compute a disagreement rate per pairing strategy -- e.g. what fraction of `keeper_removed` pairs the judge thinks *aren't* actually duplicates (candidate wrong removals), or what fraction of `cross_group_sample` pairs the judge thinks *are* duplicates (candidate missed duplicates). It writes every disagreeing pair to a JSONL file for manual review, since a disagreement is a signal to look at the pair yourself, not a confirmed fuzzy-dedup error.

By default, analysis reads the saved decision unchanged: `--decision-source main` selects the original judge result, while `--decision-source final` selects `final_decision` from step 6. Both modes export supporting fields such as materiality, confidence and risk from the selected decision, alongside the original main result and any critic audit fields. Use these default modes to compare main and the enabled critics on the same records.

For a separate main-only diagnostic report, `--apply-main-corrections` enables two analysis rules: (1) classify complete, untruncated, nonempty, strictly equal original texts as `exact`; (2) reclassify `containment` as `related_non_duplicate` when main reports mismatched content profiles, both sides `non_main_only`, or no shared basis. Zero unique spans alone do not prove raw-text equality because alignment normalizes tokens. Corrections are counted in the logs; disagreement exports retain `relation_type_raw` and the flags `relation_type_corrected` / `relation_type_containment_corrected`. This option changes only the analysis report, not saved judgments, and cannot be combined with `--decision-source final`. Do not compare a corrected main report with a raw final report to measure coverage's effect.

**Prerequisites:** step 1 needs network access to Common Crawl; steps 2-3 need the GPU stages (RAPIDS/cuGraph) that `FuzzyDeduplicationWorkflow` normally uses for MinHash/LSH/connected components; step 4 is CPU-only; steps 5 and 6 need GPU(s) to serve the judge model. Run the scripts with Curator and its evaluation dependencies installed. Step 6 imports its neighboring `critics` package and distributes it to Ray workers; no tutorial-specific `PYTHONPATH` is needed.

## Quick start

```bash
# Step 1: download & extract a Common Crawl sample.
python tutorials/eval/dedup/1_data_prep.py \
  --download-dir output/dedup_eval/cc_warcs \
  --output-path output/dedup_eval/raw_corpus

# Step 2: fuzzy dedup identification.
python tutorials/eval/dedup/2_run_fuzzy_dedup.py \
  --input-path output/dedup_eval/raw_corpus \
  --input-filetype jsonl \
  --cache-dir output/dedup_eval/fuzzy_cache \
  --output-dir output/dedup_eval/fuzzy_ids

# Step 3: build the labeled pair dataset. Choose one --pair-strategy (see
# above), and make sure --input-path/--input-filetype/--input-blocksize
# match step 2 exactly.
python tutorials/eval/dedup/3_build_pair_dataset.py \
  --input-path output/dedup_eval/raw_corpus \
  --input-filetype jsonl \
  --cache-dir output/dedup_eval/fuzzy_cache \
  --fuzzy-output-dir output/dedup_eval/fuzzy_ids \
  --output-path output/dedup_eval/keeper_removed_pairs_raw \
  --pair-strategy keeper_removed

# Step 4: add span-alignment evidence to each pair.
python tutorials/eval/dedup/4_span_alignment.py \
  --input-path output/dedup_eval/keeper_removed_pairs_raw \
  --output-path output/dedup_eval/keeper_removed_pairs

# Step 5: judge each pair. First edit judge_config/fuzzy_pair_judge.yaml --
# set models[0].model to a local model path or HF repo id, and size
# num_replicas/tensor_parallel_size to your GPUs.
python tutorials/eval/dedup/5_run_llm_judge.py \
  --input-path output/dedup_eval/keeper_removed_pairs \
  --output-path output/dedup_eval/judged_pairs

# Step 6 (optional): review saved main results with coverage.
# Add --subject to also run the subject critic and its verifier.
python tutorials/eval/dedup/6_run_critics.py \
  --input-path output/dedup_eval/judged_pairs \
  --output-path output/dedup_eval/reviewed_pairs

# Step 7: compare the saved main and final decisions on the same records.
# If step 6 was skipped, use judged_pairs and --decision-source main only.
python tutorials/eval/dedup/7_analyze_results.py \
  --judge-output-path output/dedup_eval/reviewed_pairs --decision-source main \
  --disagreements-output output/dedup_eval/main_disagreements.jsonl
python tutorials/eval/dedup/7_analyze_results.py \
  --judge-output-path output/dedup_eval/reviewed_pairs --decision-source final \
  --disagreements-output output/dedup_eval/final_disagreements.jsonl

# Optional: a separate report with main-only analysis corrections.
python tutorials/eval/dedup/7_analyze_results.py \
  --judge-output-path output/dedup_eval/reviewed_pairs --apply-main-corrections \
  --disagreements-output output/dedup_eval/corrected_main_disagreements.jsonl
```

## Critics

Each critic lives in a tutorial module; `critics/coverage.py` owns coverage's routing, schema and rules, and `critics/subject.py` owns subject proposals and verification, while `critics/stages.py` prepares and applies records. Step 6 defaults to source judge `pair_semantic_judgment` and the YAML's first model alias; override them with `--source-judge` and `--model-alias`. Prompts live in `judge_config/critics/coverage/` and `judge_config/critics/subject/`. Main results must contain the eight decision fields required by `critics/dedup_adapter.py`. Unique spans need `delta_start_char` / `delta_end_char` to separate core evidence (at most 240 characters) from padded reading context; incompatible inputs fail validation.

Coverage skips negative/unresolved main decisions and complete, untruncated, nonempty, strictly equal texts. Other positive replacement decisions are reviewed using evidence without the main prediction or pair label. Python validates references and applies keep, directional veto or abstention rules. A veto can only revoke an existing positive direction and requires complete, untruncated evidence. Missing required reviews or invalid references fail validation rather than counting as skips.

Original fields and the main result are preserved. Step 6 adds `coverage_should_run`, `coverage_review` (null when skipped), `coverage_action`, `coverage_reason`, `coverage_evidence` and the eight-field `final_decision`; temporary payloads are removed. Check input/output IDs when accepting a run, since an apply stage cannot detect records lost upstream. Routing counts describe rows, not HTTP request counts, which may include retries. `--decision-source final` requires `final_decision` and never overrides it.

With `--subject`, the same Pipeline continues after coverage: subject prepare → NDD subject proposal → Python proof validation → NDD verifier → Python final application. Both model columns use native `SkipConfig`; all calls share step 6's existing inference service and selected model. The default remains coverage only.

Subject review requires a remaining positive replacement direction, different original texts, unique spans on both sides and untruncated evidence. The seven-field proposal selects subjects and predicates by span ID. Only `LIABILITY_PARTY`, `POLICY_SERVICE` and `FAILED_OBJECT` have veto authority; `ACCESS_TARGET` and `RECORD_SUBJECT` objections are retained for audit but cannot veto. The verifier receives the original texts and Python-resolved fixed quotes, without main/coverage predictions or labels. Only `SUPPORTED_DIFFERENT_NAMED_TARGETS` with two `NAMED_ACTUAL_TARGET` classifications can revoke the remaining positive directions. Unsupported or uncertain verification keeps coverage. Malformed references, missing required reviews and conclusive proposals on incomplete packets fail validation.

Subject adds `subject_base_decision` (the coverage result), `subject_should_run`, `subject_review`, `subject_verifier_should_run`, `subject_verifier_review`, `subject_action`, `subject_reason` and `subject_evidence`. `final_decision` is updated only after verification; all main and coverage audit columns remain unchanged. Skipped proposal/verifier cells are null and temporary payloads are removed. Step 7 includes subject audit fields in disagreement exports and reads the last enabled critic's result in `final` mode. Compatibility covers the v0.7.1 subject-scope and fixed-verifier rules, not historical main arbitration.

## Files

- `1_data_prep.py` -- downloads and extracts a Common Crawl sample with jusText.
- `2_run_fuzzy_dedup.py` -- runs `FuzzyDeduplicationWorkflow(perform_removal=False)`.
- `3_build_pair_dataset.py` -- builds labeled document pairs from fuzzy dedup's groups and removal decisions.
- `4_span_alignment.py` -- diffs each pair's visible text into `SHARED`/`A_ONLY`/`B_ONLY` spans for the judge.
- `judge_config/fuzzy_pair_judge.yaml`, `judge_config/system.jinja`, `judge_config/pair.jinja` -- the LLM judge config for step 5, a semantic-retention rubric.
- `5_run_llm_judge.py` -- runs `LLMJudgeWorkflow` over the span-aligned pairs using this example's judge config.
- `6_run_critics.py`, `critics/`, `judge_config/critics/` -- optional coverage and verified subject review of saved main results.
- `7_analyze_results.py` -- buckets main or final verdicts and reports disagreement rates against fuzzy dedup's decisions.
