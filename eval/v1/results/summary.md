Questions: 49 (combined 4, cypher 12, sql 15, unanswerable 4, vector 14)

**Overall (answerable questions; refusals on unanswerable ones)**

|                          |   agent |   baseline |
|:-------------------------|--------:|-----------:|
| Route                    |    0.96 |     nan    |
| Correct                  |    0.71 |       0.35 |
| Values                   |    0.78 |       0.35 |
| Entities                 |    0.77 |       0.29 |
| Wrong refusals           |    0.09 |       0.22 |
| Cited                    |    0.80 |       0.80 |
| Source passage retrieved |    0.57 |       0.57 |
| Correct refusals         |    1.00 |       1.00 |
| Invalid citations        |    0.02 |       0.00 |
| Mean seconds             |   12.37 |       4.09 |

**By category**

| category   | system   |   Route |   Correct |   Values |   Entities |   Wrong refusals |   Cited |   Source passage retrieved |
|:-----------|:---------|--------:|----------:|---------:|-----------:|-----------------:|--------:|---------------------------:|
| combined   | agent    |    0.75 |      0.25 |     0.38 |       0.67 |             0.50 |    0.50 |                     nan    |
| combined   | baseline |  nan    |      0.25 |     0.25 |       0.00 |             0.50 |    0.25 |                     nan    |
| cypher     | agent    |    1.00 |      0.67 |     1.00 |       0.73 |             0.08 |    0.50 |                     nan    |
| cypher     | baseline |  nan    |      0.33 |     0.00 |       0.45 |             0.17 |    0.83 |                     nan    |
| sql        | agent    |    1.00 |      0.87 |     0.87 |       1.00 |             0.00 |    1.00 |                     nan    |
| sql        | baseline |  nan    |      0.40 |     0.40 |       0.00 |             0.40 |    0.73 |                     nan    |
| vector     | agent    |    0.93 |    nan    |   nan    |     nan    |             0.07 |    0.93 |                       0.57 |
| vector     | baseline |  nan    |    nan    |   nan    |     nan    |             0.00 |    1.00 |                       0.57 |
