# Scoring tables

`ippt_tables.csv` contains the official IPPT scoring lookup for push-ups and
sit-ups (male), sourced from the "New IPPT Format And Scoring System For The
Home Team" tables via [ninest/ippt-utils](https://github.com/ninest/ippt-utils).

The same tables are used by SAF, HTA (MHA), and SPF.

## Format

```csv
exercise,sex,age_min,age_max,reps,points
```

- `exercise`: `pushup` or `situp`
- `sex`: `m` (male; female tables not yet added)
- `age_min`, `age_max`: inclusive 3-year age band (0–21, 22–24, …, 58–60)
- `reps`: valid repetitions in one minute (1–60)
- `points`: points awarded (0–25)

The scorer uses the highest listed `reps` that does not exceed the participant's
count (0 below the lowest listed row, the top row's points above the highest).

## Award tiers (from total score across all 3 stations)

| Tier              | Total points |
|-------------------|-------------|
| Gold              | ≥ 85        |
| Silver            | 75–84       |
| Pass w/ Incentive | 61–74       |
| Pass              | 51–60       |
| Fail              | ≤ 50 **or** 0 in any station |

## Zero-score rule

A score of 0 in any station is an automatic fail regardless of total points.

## Source and verification

Data extracted from `ippt-utils` v0.1.4 (MIT licence). The package notes
"All score tables are from New IPPT Format And Scoring System For The Home Team."
Cross-check against the [MHA NS Portal IPPT tools](https://ns.mha.gov.sg/nsp/app/ns-and-you/ippt-and-ns-fit)
or the CMPB station PDFs before using for anything official.

Last checked: October 2026.
