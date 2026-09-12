# Retention-v5 completed review: rejected

Run: `/raid/hfang/ihb/runs/critic-five-hour-20260911T070052Z/retention-v5-paired96`.
354 valid HTTP 200 responses, no retries; all 96 original rows and both repeats.
No main calls, label changes or release. All 50 cross-arm error IDs are reviewed.

| Arm | Repeat | Weighted precision | Recall | Primary exact |
| --- | --- | --- | --- | --- |
| Fresh original .12 | 1 | 74.73% | 45.21% | 70.91% |
| Fresh original .12 | 2 | 68.20% | 43.99% | 68.57% |
| Exact strict v4 | 1 | 59.87% | 85.66% | 70.41% |
| Exact strict v4 | 2 | 61.97% | 85.66% | 72.28% |
| Strict v5 | 1 | 59.87% | 85.66% | 68.40% |
| Strict v5 | 2 | 58.96% | 85.66% | 67.53% |

V5 loses H0232/H0312 twice despite explaining that the labels do not contain facts.
Actual forum subjects H0537/H0850 fail twice; H0606 and H0907 fluctuate across repeats.
H0606 repeat 2 happens to give NO through two losses while still incorrectly denying
the liability binding. Correct final output alone does not prove correct reasoning.
Fresh v4 also fails H0850 once; earlier protection was not universally stable.

H0533 is a new functional-UI/content question: an access-icon legend versus a citation
handle, copyright and deletion-confirmation prompt. Neither an actual deletion nor a
specific current access permission should be invented from these generic UI elements.
The old NO reference is retained pending adjudication; no hard guard is added from it.
H0514 actual profiles/team-roster access target is a clear legacy error, not such a dispute.

New experiment keeps v4 semantics unchanged and moves only its explanation property
before the proof decisions in the strict schema. This isolates a generation-order
hypothesis: every inspected v3/v4/v5 output selected evidence/conflict before explanation.
The request artifact's canonical object sorting preserved the unwanted schema order.
The new order-preserving request string is separately hashed and tested through a real
local HTTP client and pacing relay. Actual NVIDIA output order must still be verified;
no semantic benefit is claimed before observing paired results.

Selected-material P/R never replaces full original development scores. The best verified
balanced full baseline remains .12 final, P 67.99%, R 75.07% on the unchanged partial draft.
