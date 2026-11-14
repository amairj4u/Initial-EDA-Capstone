# Early Intervention Claims Triaging System

Capstone project: predicting high-cost workers compensation claims from
early-stage claims data.

## Research question

Can early-stage workers compensation claims data reliably predict which claims
will escalate into high-cost outcomes, so they can be flagged as high risk or
low risk soon after they are filed?

## Notebook

The full analysis is in the Jupyter notebook:

[notebooks/eda_and_baseline_model.ipynb](notebooks/eda_and_baseline_model.ipynb)

It covers data cleaning, feature engineering, exploratory data analysis with
visualizations, and one baseline machine learning model (Logistic Regression)
that classifies each claim as high risk or low risk.

## Data

The project uses the Actuarial Loss Prediction dataset (workers compensation
claims). It contains 54,000 claims. Each row is a single claim with claimant
attributes, job attributes, a short text description of the injury, the insurer
initial cost estimate, and the final total claim cost (UltimateIncurredClaimCost).

The dataset is included in this project at data/train.csv, so the notebook runs
without any extra steps.

Download link (direct CSV, no login required):
https://raw.githubusercontent.com/chihangs/insurance_claims/main/train.csv

This is a public mirror of the dataset. The original source is the Kaggle
Actuarial Loss Estimation competition
(https://www.kaggle.com/competitions/actuarial-loss-estimation), whose data page
requires a free Kaggle account. The direct link above provides the same train.csv
without a login. To refresh the file, download it from the link and save it as
data/train.csv.

There is no ready-made high risk column, so the target is engineered by
thresholding the final claim cost at the 90th percentile. Claims above the
threshold are labelled high risk. The cost columns are removed from the model
inputs to avoid leakage.

## Methods

- Data cleaning: removal of duplicate rows, conversion of date columns,
  imputation of invalid ages and missing marital status.
- Outlier analysis: interquartile range review of the continuous variables, with
  a reasoned decision to keep genuine high cost claims and winsorize extreme
  weekly wages.
- Feature engineering: reporting lag, accident timing, total dependents, claim
  description word count and body part, and severe injury keyword flags.
- Exploratory data analysis: count and bar plots for categorical variables,
  histograms and boxplots for continuous variables, an interactive Plotly bar
  chart, and a correlation heatmap.
- Baseline model: Logistic Regression using early features only.

## Evaluation metric

The primary metric is recall on the high risk class, supported by the area under
the ROC curve (AUC). The data is imbalanced, so accuracy is misleading, and in
early intervention a missed high risk claim (false negative) is more costly than a
false alarm (false positive). Recall measures how many truly high risk claims the
model catches, and AUC measures how well it ranks claims by risk regardless of the
threshold.

## Summary of findings

Exploratory data analysis:

- The final claim cost is heavy tailed. About 10 percent of claims account for
  the expensive outcomes, which is the group the triage system needs to flag.
- The strongest early signals of a high risk claim are the presence of a severe
  injury keyword in the claim description, higher weekly wages, and a longer
  delay between the accident and the report.
- Full time status and older age are associated with a modestly higher risk.

Baseline model (Logistic Regression, early features only):

- ROC AUC of about 0.73 on the held-out test set, above the random baseline of
  0.5.
- With balanced class weights the model recalls about 65 percent of true high
  risk claims, which is the priority for early intervention, at the cost of some
  false positives.
- These are baseline results. The next module will test whether other models
  improve on them.

Conclusion: early-stage claim data does carry usable signal for flagging high
risk workers compensation claims soon after they are filed, and a simple model
already captures a meaningful part of it.

## Next steps

Planned for the next module:

- Compare additional models (Decision Trees, K Nearest Neighbors, and Support
  Vector Machines) against this baseline.
- Tune the decision threshold to balance catching high risk claims against false
  alarms.
- Add richer text features and test the effect of including the insurer initial
  estimate.

## Project structure

```
workers-comp-claims-triage/
  README.md
  data/
    train.csv
  notebooks/
    eda_and_baseline_model.ipynb
```

## How to run

1. Install the requirements: pandas, numpy, matplotlib, seaborn, scikit-learn,
   and jupyter.
2. Place train.csv in the data folder (the included sample works as is).
3. Open the notebook and run all cells:

```
jupyter notebook notebooks/eda_and_baseline_model.ipynb

```
