import math
from pathlib import Path
import warnings

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "bank-additional-full.csv"
MODEL_PATH = BASE_DIR / "best_model.pkl"
COLUMNS_PATH = BASE_DIR / "model_columns.pkl"

NUMERIC_FEATURES = [
    "age",
    "campaign",
    "pdays",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]
CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
YES_COLOR = "#0e9f6e"
NO_COLOR = "#94a3b8"
TEXT = "#172b43"
GRID = "#e8edf3"
MONTH_ORDER = ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]

st.set_page_config(
    page_title="Campaign Lens | Term Deposit Insights",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp { background: #ffffff; color: #1f2937; }
    section.main .block-container { padding: 2rem 2rem 3rem; max-width: 1500px; }
    .stApp [data-testid="stWidgetLabel"],
    .stApp [data-testid="stWidgetLabel"] *,
    .stApp [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stCaptionContainer"] {
        color: #243449 !important;
    }
    .stApp [data-testid="stCaptionContainer"] { color: #526174 !important; }
    .stApp [data-testid="stTextInput"] input,
    .stApp [data-testid="stNumberInput"] input,
    .stApp [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        background: #ffffff !important;
        color: #1f2937 !important;
        border-color: #cbd5e1 !important;
        -webkit-text-fill-color: #1f2937 !important;
    }
    [data-testid="stSidebar"] {
        background: #10243a;
        width: 310px !important;
        min-width: 310px !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #f1f5f9 !important;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #cbd5e1 !important;
    }
    [data-testid="stSidebar"] [data-testid="stMultiSelect"] [data-baseweb="select"] > div {
        background: #ffffff !important;
        color: #1f2937 !important;
    }
    [data-testid="stSidebar"] [data-testid="stMultiSelect"] span {
        color: #1f2937 !important;
    }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px 16px;
    }
    [data-testid="stMetricLabel"] { color: #475569; }
    [data-testid="stMetricValue"] { color: #172b43; }
    div[data-testid="stForm"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
    }
    div[data-testid="stForm"] [data-testid="stFormSubmitButton"] button {
        background: #16835d;
        border-color: #16835d;
        color: #ffffff !important;
    }
    [data-testid="stDownloadButton"] button {
        background: #16835d !important;
        border-color: #16835d !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    [data-testid="stDownloadButton"] button p,
    [data-testid="stDownloadButton"] button span {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    @media (max-width: 900px) {
        section.main .block-container { padding: 1.25rem 1rem 2rem; }
        .main .stHorizontalBlock { flex-wrap: wrap; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, sep=";")


@st.cache_resource
def load_model():
    with warnings.catch_warnings():
        warnings.filterwarnings("default", category=UserWarning)
        model = joblib.load(MODEL_PATH)
    if not hasattr(model, "multi_class"):
        # Older scikit-learn versions still access this attribute in predict_proba.
        model.multi_class = "auto"
    return model


@st.cache_data
def load_model_columns():
    return joblib.load(COLUMNS_PATH)


def build_prediction_input(values, columns, data):
    row = pd.DataFrame([values], columns=FEATURES)
    for column in CATEGORICAL_FEATURES:
        if row.at[0, column] == "unknown":
            known_values = data.loc[data[column] != "unknown", column]
            row.at[0, column] = known_values.mode().iloc[0]
    return pd.get_dummies(row).reindex(columns=columns, fill_value=0)


def page_header(kicker, title, subtitle):
    st.caption(kicker)
    st.title(title)
    st.write(subtitle)


def style_chart(fig, height=310):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(t=10, b=10, l=10, r=10),
        font=dict(size=12, color=TEXT),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        legend_title_text="",
    )
    fig.update_xaxes(showline=False, zeroline=False, gridcolor=GRID)
    fig.update_yaxes(showline=False, zeroline=False, gridcolor=GRID)
    return fig


def render_chart_card(title, fig, caption=None):
    with st.container():
        st.markdown(f"#### {title}")
        if caption:
            st.caption(caption)
        st.plotly_chart(fig, use_container_width=True, theme=None)


def grouped_subscription_rate(data, column, order=None):
    grouped = (
        data.assign(subscribed=data["y"].eq("yes"))
        .groupby(column, observed=True, as_index=False)
        .agg(rate=("subscribed", "mean"), customers=("subscribed", "size"))
    )
    if order:
        grouped[column] = pd.Categorical(grouped[column], order, ordered=True)
        return grouped.sort_values(column)
    return grouped.sort_values("rate", ascending=False)


def render_kpi_cards(data):
    total = len(data)
    successful = int(data["y"].eq("yes").sum())
    subscription_rate = successful / total if total else 0
    average_calls = data["campaign"].mean() if total else 0
    average_age = data["age"].mean() if total else 0

    metrics = [
        ("Total Customers", f"{total:,}"),
        ("Successful Subscriptions", f"{successful:,}"),
        ("Subscription Rate", f"{subscription_rate:.1%}"),
        ("Average Campaign Calls", f"{average_calls:.1f}"),
        ("Average Customer Age", f"{average_age:.0f}"),
    ]
    columns = st.columns(5, gap="small")
    for column, (label, value) in zip(columns, metrics):
        with column:
            with st.container(border=True):
                st.metric(label, value)


def render_overview(data):
    page_header(
        "Campaign performance",
        "Term Deposit Insights",
        "Explore customer behaviour and discover subscription trends across the bank's direct marketing campaign.",
    )

    total = len(data)
    successful = int(data["y"].eq("yes").sum())
    subscription_rate = successful / total if total else 0
    render_kpi_cards(data)
    st.caption(f"Showing {total:,} customers in the current filtered view.")

    if data.empty:
        st.info("No customers match the selected filters. Adjust them in the sidebar.")
        return

    left, right = st.columns([0.95, 1.25], gap="medium")
    with left:
        st.subheader("Subscription Outcome")
        outcomes = (
            data["y"]
            .value_counts()
            .reindex(["yes", "no"], fill_value=0)
            .rename_axis("Outcome")
            .reset_index(name="Customers")
        )
        outcomes["Outcome"] = outcomes["Outcome"].map({"yes": "Yes", "no": "No"})
        fig = px.pie(
            outcomes,
            names="Outcome",
            values="Customers",
            hole=0.72,
            color="Outcome",
            color_discrete_map={"Yes": YES_COLOR, "No": NO_COLOR},
        )
        fig.update_traces(
            textinfo="percent",
            textposition="inside",
            textfont=dict(size=14, color="#ffffff"),
            hovertemplate="%{label}: %{value:,} customers (%{percent})<extra></extra>",
        )
        fig.update_layout(
            annotations=[
                dict(
                    text=f"<b>{subscription_rate:.1%}</b><br><span style='font-size:11px'>subscribed</span>",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=22, color=TEXT),
                )
            ],
        )
        render_chart_card("Subscription Outcome", style_chart(fig, 290))

    with right:
        by_job = grouped_subscription_rate(data, "job").sort_values("rate")
        fig = px.bar(
            by_job,
            x="rate",
            y="job",
            orientation="h",
            text=by_job["rate"].map(lambda rate: f"{rate:.1%}"),
            labels={"rate": "Subscription rate", "job": "Job"},
            color_discrete_sequence=[YES_COLOR],
        )
        fig.update_traces(
            textposition="outside",
            cliponaxis=False,
            customdata=by_job[["customers"]].to_numpy(),
            hovertemplate="%{y}<br>Subscription rate: %{x:.1%}<br>Customers: %{customdata[0]:,}<extra></extra>",
        )
        fig.update_xaxes(tickformat=".0%", range=[0, by_job["rate"].max() * 1.18], title_text=None)
        fig.update_yaxes(title_text=None, showgrid=False)
        render_chart_card("Subscription Rate by Job", style_chart(fig, 290))

    education, marital = st.columns(2, gap="medium")
    with education:
        by_education = grouped_subscription_rate(data, "education").sort_values("rate")
        fig = px.bar(
            by_education,
            x="rate",
            y="education",
            orientation="h",
            text=by_education["rate"].map(lambda rate: f"{rate:.1%}"),
            labels={"rate": "Subscription rate", "education": "Education"},
            color_discrete_sequence=[YES_COLOR],
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_xaxes(tickformat=".0%", range=[0, by_education["rate"].max() * 1.2], title_text=None)
        fig.update_yaxes(title_text=None, showgrid=False)
        render_chart_card("Subscription by Education", style_chart(fig, 330))

    with marital:
        by_marital = grouped_subscription_rate(data, "marital").sort_values("rate")
        fig = px.bar(
            by_marital,
            x="rate",
            y="marital",
            orientation="h",
            text=by_marital["rate"].map(lambda rate: f"{rate:.1%}"),
            labels={"rate": "Subscription rate", "marital": "Marital status"},
            color_discrete_sequence=[YES_COLOR],
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_xaxes(tickformat=".0%", range=[0, by_marital["rate"].max() * 1.2], title_text=None)
        fig.update_yaxes(title_text=None, showgrid=False)
        render_chart_card("Subscription by Marital Status", style_chart(fig, 330))

    month_column, age_column = st.columns(2, gap="medium")
    with month_column:
        monthly = grouped_subscription_rate(data, "month", MONTH_ORDER)
        fig = px.bar(
            monthly,
            x="month",
            y="rate",
            labels={"month": "Contact month", "rate": "Subscription rate"},
            color_discrete_sequence=[YES_COLOR],
        )
        fig.update_traces(
            hovertemplate="Month: %{x}<br>Subscription rate: %{y:.1%}<extra></extra>",
            text=monthly["rate"].map(lambda rate: f"{rate:.1%}"),
            textposition="outside",
        )
        fig.update_yaxes(tickformat=".0%", title_text=None, range=[0, monthly["rate"].max() * 1.2])
        render_chart_card("Subscription by Contact Month", style_chart(fig, 320))

    with age_column:
        age_data = data.copy()
        age_data["age_group"] = pd.cut(
            age_data["age"],
            bins=[0, 25, 35, 45, 55, 65, float("inf")],
            labels=["Under 25", "25–34", "35–44", "45–54", "55–64", "65+"],
            right=False,
        )
        age_order = ["Under 25", "25–34", "35–44", "45–54", "55–64", "65+"]
        age_rates = grouped_subscription_rate(age_data, "age_group", age_order)
        fig = px.bar(
            age_rates,
            x="age_group",
            y="rate",
            labels={"age_group": "Customer age", "rate": "Subscription rate"},
            color_discrete_sequence=[YES_COLOR],
        )
        fig.update_traces(
            hovertemplate="Age: %{x}<br>Subscription rate: %{y:.1%}<extra></extra>",
            text=age_rates["rate"].map(lambda rate: f"{rate:.1%}"),
            textposition="outside",
        )
        fig.update_yaxes(tickformat=".0%", title_text=None, range=[0, age_rates["rate"].max() * 1.2])
        render_chart_card("Subscription by Age Group", style_chart(fig, 320))

    channel_column, _ = st.columns(2, gap="medium")
    with channel_column:
        channel = grouped_subscription_rate(data, "contact").sort_values("rate")
        fig = px.bar(
            channel,
            x="rate",
            y="contact",
            orientation="h",
            text=channel["rate"].map(lambda rate: f"{rate:.1%}"),
            labels={"contact": "Contact channel", "rate": "Subscription rate"},
            color_discrete_sequence=[YES_COLOR],
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_xaxes(tickformat=".0%", range=[0, channel["rate"].max() * 1.2], title_text=None)
        fig.update_yaxes(title_text=None, showgrid=False)
        render_chart_card("Subscription by Contact Channel", style_chart(fig, 250))

    st.caption("Historical campaign associations are descriptive and do not imply causation.")


def render_prediction(data, model, columns):
    page_header(
        "Individual estimate",
        "Predict Customer Subscription",
        "Enter a customer's profile and campaign context to estimate their term deposit subscription likelihood.",
    )
    st.info(
        "Predictions use your trained Logistic Regression model. Call duration is intentionally excluded "
        "because it is only known after the call."
    )
    st.caption(
        "If a categorical answer is “unknown”, the model input uses that field’s most common known value."
    )

    numeric_defaults = data[NUMERIC_FEATURES].median()
    numeric_ranges = data[NUMERIC_FEATURES].agg(["min", "max"])

    with st.form("prediction_form"):
        profile, campaign = st.columns(2, gap="large")
        with profile:
            st.markdown("### Customer Information")
            st.caption("Demographics and existing credit products")
            age = st.number_input(
                "Age",
                min_value=int(numeric_ranges.at["min", "age"]),
                max_value=int(numeric_ranges.at["max", "age"]),
                value=int(numeric_defaults["age"]),
            )
            job = st.selectbox("Job", sorted(data["job"].unique()))
            marital = st.selectbox("Marital status", sorted(data["marital"].unique()))
            education = st.selectbox("Education", sorted(data["education"].unique()))
            default = st.selectbox("Credit in default", sorted(data["default"].unique()))
            housing = st.selectbox("Housing loan", sorted(data["housing"].unique()))
            loan = st.selectbox("Personal loan", sorted(data["loan"].unique()))

        with campaign:
            st.markdown("### Campaign Context")
            st.caption("Contact details and economic indicators")
            contact = st.selectbox("Contact type", sorted(data["contact"].unique()))
            month = st.selectbox("Contact month", sorted(data["month"].unique()))
            day_of_week = st.selectbox("Day of week", sorted(data["day_of_week"].unique()))
            poutcome = st.selectbox("Previous campaign outcome", sorted(data["poutcome"].unique()))

            st.markdown("#### Campaign and economic indicators")
            numeric_values = {}
            for index in range(0, len(NUMERIC_FEATURES), 2):
                feature_columns = st.columns(2)
                for offset, feature in enumerate(NUMERIC_FEATURES[index : index + 2]):
                    with feature_columns[offset]:
                        lower = float(numeric_ranges.at["min", feature])
                        upper = float(numeric_ranges.at["max", feature])
                        is_integer = feature in {"campaign", "pdays", "previous"}
                        step = 1.0 if is_integer else 0.1
                        label = feature.replace(".", " ").replace("_", " ").title()
                        numeric_values[feature] = st.number_input(
                            label,
                            min_value=int(lower) if is_integer else lower,
                            max_value=int(upper) if is_integer else upper,
                            value=int(numeric_defaults[feature]) if is_integer else float(numeric_defaults[feature]),
                            step=int(step) if is_integer else step,
                            format="%.3f" if not is_integer else "%d",
                            help=f"Observed range: {lower:g} to {upper:g}.",
                        )

        submitted = st.form_submit_button(
            "Predict Subscription", type="primary", use_container_width=True
        )

    if submitted:
        values = {
            **numeric_values,
            "job": job,
            "marital": marital,
            "education": education,
            "default": default,
            "housing": housing,
            "loan": loan,
            "contact": contact,
            "month": month,
            "day_of_week": day_of_week,
            "poutcome": poutcome,
        }
        model_input = build_prediction_input(values, columns, data)
        probabilities = model.predict_proba(model_input)[0]
        positive_index = list(model.classes_).index(1)
        probability = float(probabilities[positive_index])
        prediction = int(model.predict(model_input)[0])

        st.divider()
        result, details = st.columns(2)
        with result:
            st.metric("Subscription probability", f"{probability:.1%}")
            if prediction == 1:
                st.success("Prediction: likely to subscribe")
            else:
                st.info("Prediction: unlikely to subscribe")
        with details:
            st.subheader("Result")
            st.progress(probability, text=f"Probability: {probability:.1%}")
            st.caption(
                "The classifier's decision uses a 50% probability threshold. This model estimate is "
                "decision support, not a guarantee or a substitute for campaign judgment."
            )
        with st.expander("Review the submitted profile"):
            st.dataframe(
                pd.DataFrame([values]).rename(
                    columns={
                        "emp.var.rate": "employment variation rate",
                        "cons.price.idx": "consumer price index",
                        "cons.conf.idx": "consumer confidence index",
                        "euribor3m": "Euribor 3-month rate",
                        "nr.employed": "number employed",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )


def render_explorer(data):
    page_header(
        "Campaign records",
        "Customer Explorer",
        "Search and filter customer records, review campaign attributes, or export the current selection.",
    )
    st.caption(f"{len(data):,} customer records · {len(data.columns) - 1} input fields")

    col1, col2, col3, col4 = st.columns([1, 1, 1, 1.5], gap="medium")
    with col1:
        outcome = st.selectbox("Subscription outcome", ["All", "yes", "no"])
    with col2:
        education = st.selectbox("Education", ["All", *sorted(data["education"].unique())])
    with col3:
        marital = st.selectbox("Marital status", ["All", *sorted(data["marital"].unique())])
    with col4:
        search = st.text_input("Search job or education", placeholder="e.g. technician")

    result = data
    if outcome != "All":
        result = result[result["y"] == outcome]
    if education != "All":
        result = result[result["education"] == education]
    if marital != "All":
        result = result[result["marital"] == marital]
    if search.strip():
        query = search.strip().lower()
        result = result[
            result["job"].str.lower().str.contains(query, na=False)
            | result["education"].str.lower().str.contains(query, na=False)
        ]

    count_column, page_column, download_column = st.columns([2, 1, 1])
    page_size = page_column.selectbox("Rows per page", [25, 50, 100], index=0)
    page_count = max(1, math.ceil(len(result) / page_size))
    if st.session_state.get("explorer_page", 1) > page_count:
        st.session_state["explorer_page"] = 1
    page_number = count_column.number_input(
        f"Page (1–{page_count})",
        min_value=1,
        max_value=page_count,
        step=1,
        key="explorer_page",
    )
    start = (int(page_number) - 1) * page_size
    first_row = min(start + 1, len(result)) if len(result) else 0
    last_row = min(start + page_size, len(result))
    st.caption(f"Showing {first_row:,}–{last_row:,} of {len(result):,} matching records")
    download_column.download_button(
        "Download filtered CSV",
        data=result.to_csv(index=False).encode("utf-8"),
        file_name="bank_campaign_filtered.csv",
        mime="text/csv",
        use_container_width=True,
        disabled=result.empty,
    )
    visible_columns = [
        "age",
        "job",
        "marital",
        "education",
        "contact",
        "month",
        "campaign",
        "poutcome",
        "y",
    ]
    display_data = result.iloc[start : start + page_size][visible_columns].rename(
        columns={
            "age": "Age",
            "job": "Job",
            "marital": "Marital status",
            "education": "Education",
            "contact": "Contact",
            "month": "Month",
            "campaign": "Campaign calls",
            "poutcome": "Previous outcome",
            "y": "Subscribed",
        }
    )
    st.dataframe(
        display_data,
        use_container_width=True,
        hide_index=True,
        height=520,
    )


def main():
    required_files = [DATA_PATH, MODEL_PATH, COLUMNS_PATH]
    missing_files = [path.name for path in required_files if not path.is_file()]
    if missing_files:
        st.error(f"Required project files are missing: {', '.join(missing_files)}")
        st.stop()

    data = load_data()
    model = load_model()
    columns = load_model_columns()
    if not set(FEATURES + ["y"]).issubset(data.columns):
        st.error("The dataset is missing one or more expected model or target fields.")
        st.stop()
    if len(columns) != model.n_features_in_:
        st.error("The model feature list does not match the saved model. Check the model artifacts.")
        st.stop()

    with st.sidebar:
        st.subheader("Campaign Lens")
        st.caption("Bank Marketing | 2008–2010")
        page = st.radio(
            "Navigate",
            ["Overview", "Predict", "Customer Explorer"],
            label_visibility="collapsed",
            key="active_page",
        )
        if page != "Predict":
            st.divider()
            st.markdown("### Refine dashboard")
            st.caption("Filters apply to the overview and customer records.")
            if st.button("Reset filters", use_container_width=True, key="reset_campaign_filters"):
                for key in ["filter_jobs", "filter_months", "filter_channels"]:
                    st.session_state.pop(key, None)

            job_options = sorted(data["job"].dropna().unique())
            month_options = [month for month in MONTH_ORDER if month in set(data["month"])]
            channel_options = sorted(data["contact"].dropna().unique())
            selected_jobs = st.multiselect(
                "Job",
                job_options,
                default=job_options,
                key="filter_jobs",
            )
            selected_months = st.multiselect(
                "Contact month",
                month_options,
                default=month_options,
                key="filter_months",
            )
            selected_channels = st.multiselect(
                "Contact channel",
                channel_options,
                default=channel_options,
                key="filter_channels",
            )
        else:
            selected_jobs = data["job"].dropna().unique().tolist()
            selected_months = MONTH_ORDER
            selected_channels = data["contact"].dropna().unique().tolist()
        st.divider()
        st.caption("UCI Bank Marketing · Classification model")

    filtered_data = data[
        data["job"].isin(selected_jobs)
        & data["month"].isin(selected_months)
        & data["contact"].isin(selected_channels)
    ]
    if page == "Overview":
        render_overview(filtered_data)
    elif page == "Predict":
        render_prediction(data, model, columns)
    else:
        render_explorer(filtered_data)


if __name__ == "__main__":
    main()
