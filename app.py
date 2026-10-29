from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"

st.set_page_config(
    page_title="Financial Time-Series Forecasting",
    page_icon="📈",
    layout="wide",
)

st.title("Financial Time-Series Forecasting with Deep Learning")
st.caption("Academic project dashboard · GRU evaluation, feature ablation and cross-market generalization")
st.warning(
    "Research demonstration only. Forecast accuracy does not imply trading profitability. "
    "This dashboard is not investment advice."
)

@st.cache_data
def read_csv(filename):
    path = DATA_DIR / filename
    if not path.exists():
        return None
    return pd.read_csv(path)

exp2_summary = read_csv("exp2_walk_forward_summary.csv")
exp2_folds = read_csv("exp2_walk_forward_fold_metrics.csv")
exp2_lookback = read_csv("exp2_lookback_validation.csv")
exp3_features = read_csv("exp3_feature_validation_metrics.csv")
exp4_metrics = read_csv("exp4_test_metrics.csv")
exp4_yearly = read_csv("exp4_yearly_metrics.csv")
exp4_predictions = read_csv("exp4_test_predictions.csv")

pages = [
    "Overview",
    "Experiment 2 — Walk-forward",
    "Experiment 3 — Feature ablation",
    "Experiment 4 — NIFTY Bank",
    "Methodology & limitations",
]
page = st.sidebar.radio("Navigate", pages)
st.sidebar.markdown("---")
st.sidebar.caption("Built with Python, pandas, Plotly and Streamlit.")


def find_col(df, required_terms, exclude_terms=()):
    if df is None:
        return None
    for col in df.columns:
        lower = str(col).lower()
        if all(term.lower() in lower for term in required_terms) and not any(
            term.lower() in lower for term in exclude_terms
        ):
            return col
    return None


def metric_row(df, model_name):
    if df is None or "Model" not in df.columns:
        return None
    rows = df[df["Model"].astype(str).str.casefold() == model_name.casefold()]
    return None if rows.empty else rows.iloc[0]


if page == "Overview":
    st.header("Project overview")
    st.write(
        "The project compares recurrent neural networks and simple forecasting baselines, "
        "tests added input features, and evaluates whether a GRU generalizes to a second market index."
    )
    if exp4_metrics is not None and "Model" in exp4_metrics.columns:
        gru = metric_row(exp4_metrics, "GRU")
        zero = metric_row(exp4_metrics, "Zero-return baseline")
        if gru is not None and zero is not None:
            rmse_col = find_col(exp4_metrics, ["Return RMSE"])
            mae_col = find_col(exp4_metrics, ["Return MAE"])
            close_rmse_col = find_col(exp4_metrics, ["Close RMSE"])
            if rmse_col:
                delta = float(gru[rmse_col]) - float(zero[rmse_col])
                c1, c2, c3 = st.columns(3)
                c1.metric("NIFTY Bank GRU return RMSE", f"{float(gru[rmse_col]):.4f} pp",
                          f"{delta:+.4f} pp vs zero-return", delta_color="inverse")
                if mae_col:
                    c2.metric("GRU return MAE", f"{float(gru[mae_col]):.4f} pp")
                if close_rmse_col:
                    c3.metric("GRU close RMSE", f"{float(gru[close_rmse_col]):.2f} points")
                st.info(
                    "In the recorded cross-market test, the GRU's return RMSE is higher than "
                    "the zero-return baseline's. The current evidence does not show a consistent GRU advantage."
                )
        st.subheader("Cross-market test metrics")
        st.dataframe(exp4_metrics, use_container_width=True, hide_index=True)
    else:
        st.info("Experiment 4 metrics CSV is missing. Run the dashboard packaging notebook after the Experiment 4 save cell.")

    st.subheader("Experiments in this project")
    st.markdown(
        "1. **Baseline models:** compare RNN, LSTM and GRU forecasts.\n"
        "2. **Lookback sensitivity:** compare 20-, 30- and 60-session GRU input windows.\n"
        "3. **Feature ablation:** compare returns, momentum, and momentum plus volatility.\n"
        "4. **Cross-market generalization:** repeat the selected GRU method on NIFTY Bank."
    )

elif page == "Experiment 2 — Walk-forward":
    st.header("Experiment 2 — Lookback sensitivity and walk-forward validation")
    if exp2_summary is not None:
        st.subheader("Aggregate walk-forward performance")
        st.dataframe(exp2_summary, use_container_width=True, hide_index=True)
        model_col = "Model" if "Model" in exp2_summary.columns else None
        rmse_col = find_col(exp2_summary, ["Return RMSE"])
        if model_col and rmse_col:
            chart = px.bar(exp2_summary, x=model_col, y=rmse_col,
                           title="Aggregate return RMSE by model (lower is better)", text_auto=".4f")
            st.plotly_chart(chart, use_container_width=True)
    else:
        st.info("Experiment 2 aggregate CSV not found.")
    if exp2_folds is not None:
        st.subheader("Per-fold results")
        st.dataframe(exp2_folds, use_container_width=True, hide_index=True)
        fold_col = find_col(exp2_folds, ["Fold"])
        model_col = "Model" if "Model" in exp2_folds.columns else None
        rmse_col = find_col(exp2_folds, ["Return RMSE"])
        if fold_col and model_col and rmse_col:
            chart = px.line(exp2_folds, x=fold_col, y=rmse_col, color=model_col, markers=True,
                            title="Return RMSE across sequential test folds")
            st.plotly_chart(chart, use_container_width=True)
    if exp2_lookback is not None:
        st.subheader("Validation-only lookback comparison")
        st.dataframe(exp2_lookback, use_container_width=True, hide_index=True)

elif page == "Experiment 3 — Feature ablation":
    st.header("Experiment 3 — Feature-set ablation")
    if exp3_features is not None:
        st.dataframe(exp3_features, use_container_width=True, hide_index=True)
        feature_col = find_col(exp3_features, ["Feature set"])
        gru_rmse_col = find_col(exp3_features, ["GRU", "Return RMSE"])
        zero_rmse_col = find_col(exp3_features, ["Zero-return", "RMSE"])
        if feature_col and gru_rmse_col:
            cols = [feature_col, gru_rmse_col]
            if zero_rmse_col:
                cols.append(zero_rmse_col)
            chart_data = exp3_features[cols].melt(
                id_vars=[feature_col], var_name="Metric", value_name="RMSE (pp)"
            )
            chart = px.bar(chart_data, x=feature_col, y="RMSE (pp)", color="Metric", barmode="group",
                           title="GRU validation RMSE by feature set (lower is better)", text_auto=".4f")
            st.plotly_chart(chart, use_container_width=True)
        st.caption("Feature sets were compared on development validation data; the previously inspected 2024–2026 test period was not used for this selection.")
    else:
        st.info("Experiment 3 feature-validation CSV not found.")

elif page == "Experiment 4 — NIFTY Bank":
    st.header("Experiment 4 — Cross-market generalization")
    if exp4_metrics is not None:
        st.subheader("Aggregate test metrics")
        st.dataframe(exp4_metrics, use_container_width=True, hide_index=True)
        rmse_col = find_col(exp4_metrics, ["Return RMSE"])
        if "Model" in exp4_metrics.columns and rmse_col:
            chart = px.bar(exp4_metrics, x="Model", y=rmse_col,
                           title="NIFTY Bank test-period return RMSE (lower is better)", text_auto=".4f")
            st.plotly_chart(chart, use_container_width=True)
    if exp4_yearly is not None:
        st.subheader("Year-by-year performance")
        st.dataframe(exp4_yearly, use_container_width=True, hide_index=True)
        rmse_col = find_col(exp4_yearly, ["Return RMSE"])
        if "Year" in exp4_yearly.columns and "Model" in exp4_yearly.columns and rmse_col:
            yearly_chart = px.line(exp4_yearly, x="Year", y=rmse_col, color="Model", markers=True,
                                   title="Return RMSE by calendar year (lower is better)")
            st.plotly_chart(yearly_chart, use_container_width=True)
    if exp4_predictions is not None and "Date" in exp4_predictions.columns:
        st.subheader("Actual close vs one-step-ahead estimate")
        data = exp4_predictions.copy()
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        pred_cols = [c for c in data.columns if c.endswith(" close prediction")]
        if "Actual close" in data.columns and pred_cols:
            model_names = [c.removesuffix(" close prediction") for c in pred_cols]
            selected_model = st.selectbox("Forecast series", model_names, index=model_names.index("GRU") if "GRU" in model_names else 0)
            pred_col = f"{selected_model} close prediction"
            n_points = st.slider("Number of most recent test observations", min_value=60, max_value=max(60, min(500, len(data))), value=min(180, max(60, len(data))), step=30)
            plot_data = data.dropna(subset=["Date"]).tail(n_points)
            plot_long = plot_data[["Date", "Actual close", pred_col]].melt(id_vars="Date", var_name="Series", value_name="Index points")
            line = px.line(plot_long, x="Date", y="Index points", color="Series", title="Actual close and selected model's estimate")
            st.plotly_chart(line, use_container_width=True)
        else:
            st.warning("Prediction CSV does not contain the expected price columns.")
    else:
        st.info("Experiment 4 prediction CSV not found; the metrics sections remain available if their CSVs exist.")

else:
    st.header("Methodology and limitations")
    st.markdown(
        "### Evaluation design\n"
        "- Financial data is split in chronological order.\n"
        "- Scalers are fitted on training data only.\n"
        "- Lookback and feature-set selection use validation data, not the test metrics.\n"
        "- The cross-market experiment retrains the GRU from scratch on a second index.\n\n"
        "### Findings to report honestly\n"
        "- The returns-only feature set had the lowest GRU validation RMSE among the tested feature sets.\n"
        "- In the NIFTY Bank test, GRU did not consistently outperform the zero-return baseline.\n"
        "- Direction accuracy close to 50% is not evidence of statistical significance or profitability.\n\n"
        "### Limitations\n"
        "- Results depend on the selected date range, market, data source, and random seed.\n"
        "- Historical forecasting performance does not guarantee future performance.\n"
        "- This project is educational research, not a trading system or investment recommendation."
    )

st.markdown("---")
st.caption("Academic project prototype · Verify all reported results against the saved CSV outputs before submission.")
