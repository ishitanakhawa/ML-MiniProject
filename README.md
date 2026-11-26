# Bank Term Deposit Dashboard

A Streamlit dashboard for exploring the bank marketing campaign dataset and
estimating an individual customer's term deposit subscription likelihood using
the included Logistic Regression model.

## Dashboard features

- **Overview:** review dynamically calculated customer KPIs and explore
  subscription rates by job, education, marital status, contact month, contact
  channel, and age group.
- **Predict:** estimate a customer's subscription likelihood from the model's
  training features. The form excludes call duration because it is only known
  after a call.
- **Customer explorer:** filter and search records, page through results, and
  download filtered rows as CSV.

The overview and customer explorer share the sidebar filters. The saved
Logistic Regression model remains the prediction engine; no separate serialized
Decision Tree artifact is included in this project.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Keep `bank-additional-full.csv`, `best_model.pkl`, and `model_columns.pkl` next to
`app.py`. The prediction form follows the model's training schema and excludes
call duration, which is not known until after a call. The model was serialized
with scikit-learn 1.9.1, so that version is pinned in the requirements for
consistent loading.
