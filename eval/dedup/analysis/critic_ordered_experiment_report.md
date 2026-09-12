# Explanation-first result: do not adopt

The initial trial was interrupted after one upstream HTTP 500. Its 280 subsequent
local rejections were not upstream model calls. It remains an incomplete run.

The identical fresh recovery run is complete at
`/raid/hfang/ihb/runs/critic-five-hour-20260911T070052Z/retention-v4-order96-transport-v2`.
It made 355 actual attempts: 354 HTTP 200 and one bounded 429 retry; no local rejection.
All 48 cross-arm error IDs are reviewed against existing source and actual responses.

| Arm | Repeat | Weighted precision | Recall | Primary exact |
| --- | --- | --- | --- | --- |
| Fresh original .12 | 1 | 68.79% | 45.21% | 68.89% |
| Fresh original .12 | 2 | 73.73% | 43.99% | 70.44% |
| Exact v4 / old output order | 1 | 60.82% | 85.66% | 71.27% |
| Exact v4 / old output order | 2 | 60.82% | 85.66% | 71.27% |
| V4 / explanation-first | 1 | 62.14% | 85.66% | 68.54% |
| V4 / explanation-first | 2 | 60.39% | 85.66% | 69.55% |

All 113 valid candidate outputs follow the requested property order. Five more
candidate responses are rejected: three cross-side loss IDs (H0072/H0424/H0553)
and two conflicts without any unique evidence delta (H0537/H0850). No local proof
rule is relaxed and no failed answer is repaired into a valid prediction.

Weighted primary declines 2.73 and 1.72 percentage points versus exact v4. The apparent
first-repeat precision rise is not a net capability gain: missing outputs and all
denominators remain visible, and the failed-object target H0907 is missed twice.
H0606 still mistakes the inline liability subject for a header in one repeat.
Reject the factor rather than stack new proof-field or vocabulary changes onto it.

Next: exact v4 output contract and property order, changing only the original source
span presentation. The same shared/unique spans are placed in each document's actual
visible character order to test sentence-binding and negation placement directly.
No new full 1,000-case score exists yet; no selected 96-case result satisfies the goal.
