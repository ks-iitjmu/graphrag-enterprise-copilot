Questions: 48 (combined 4, cypher 11, sql 15, unanswerable 4, vector 14)

**Overall (answerable questions; refusals on unanswerable ones)**

|                          |   agent |   baseline |
|:-------------------------|--------:|-----------:|
| Route                    |    1.00 |     nan    |
| Correct                  |    0.97 |       0.33 |
| Values                   |    1.00 |       0.30 |
| Entities                 |    0.98 |       0.31 |
| Wrong refusals           |    0.00 |       0.23 |
| Cited                    |    1.00 |       0.86 |
| Source passage retrieved |    0.50 |       0.57 |
| Correct refusals         |    1.00 |       1.00 |
| Invalid citations        |    0.02 |       0.01 |
| Mean seconds             |   12.59 |       4.19 |
| answer_relevancy         |    0.96 |       0.79 |
| context_precision        |    0.82 |       0.40 |
| context_recall           |    1.00 |       0.57 |
| faithfulness             |    0.92 |       0.78 |

**By category**

| category   | system   |   Route |   Correct |   Values |   Entities |   Wrong refusals |   Cited |   Source passage retrieved |   answer_relevancy |   context_precision |   context_recall |   faithfulness |
|:-----------|:---------|--------:|----------:|---------:|-----------:|-----------------:|--------:|---------------------------:|-------------------:|--------------------:|-----------------:|---------------:|
| combined   | agent    |    1.00 |      1.00 |     1.00 |       1.00 |             0.00 |    1.00 |                     nan    |               0.89 |                0.30 |             1.00 |           0.94 |
| combined   | baseline |  nan    |      0.00 |     0.00 |       0.00 |             0.75 |    0.50 |                     nan    |               0.42 |                0.05 |             0.25 |           0.62 |
| cypher     | agent    |    1.00 |      0.91 |     1.00 |       0.97 |             0.00 |    1.00 |                     nan    |               0.96 |                1.00 |             1.00 |           0.95 |
| cypher     | baseline |  nan    |      0.36 |     0.00 |       0.49 |             0.18 |    0.82 |                     nan    |               0.72 |                0.22 |             0.36 |           0.67 |
| sql        | agent    |    1.00 |      1.00 |     1.00 |       1.00 |             0.00 |    1.00 |                     nan    |               0.97 |                1.00 |             1.00 |           0.90 |
| sql        | baseline |  nan    |      0.40 |     0.40 |       0.00 |             0.33 |    0.87 |                     nan    |               0.77 |                0.33 |             0.40 |           0.73 |
| vector     | agent    |    1.00 |    nan    |   nan    |     nan    |             0.00 |    1.00 |                       0.50 |               0.96 |                0.63 |             1.00 |           0.91 |
| vector     | baseline |  nan    |    nan    |   nan    |     nan    |             0.00 |    1.00 |                       0.57 |               0.96 |                0.71 |             1.00 |           0.96 |

RAGAS evaluations completed per system and metric (failed after retry in brackets): agent answer_relevancy 44 (0); agent context_precision 44 (0); agent context_recall 44 (0); agent faithfulness 44 (0); baseline answer_relevancy 44 (0); baseline context_precision 44 (0); baseline context_recall 44 (0); baseline faithfulness 44 (0)
