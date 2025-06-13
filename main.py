import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, accuracy_score,
                             mean_squared_error, mean_absolute_error, r2_score)
from sklearn.inspection import permutation_importance
import time

st.set_page_config(page_title="AutoML App", layout="wide", initial_sidebar_state="expanded")

st.title("AutoML App with Visual EDA and Preprocessing")

st.sidebar.header("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state["df"] = df
    st.success("Data uploaded successfully!")
else:
    st.warning("Please upload a CSV file to get started.")
    st.stop()

tabs = st.tabs(["Preview", "Data Cleaner", "EDA Summary", "Visualizer", "Model Trainer", "Insights"])

with tabs[0]:
    st.subheader("Dataset Preview")
    st.write(df.head())
    st.write(f"Shape: {df.shape}")

with tabs[1]:
    st.subheader("🧹 Data Cleaning")
    if st.checkbox("Drop Duplicates"):
        df.drop_duplicates(inplace=True)
    if st.checkbox("Drop Rows with Nulls"):
        df.dropna(inplace=True)

    st.write("Categorical Columns Encoded using LabelEncoder")
    cat_cols = df.select_dtypes(include='object').columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    if st.checkbox("Scale Numerical Features"):
        scaler = StandardScaler()
        num_cols = df.select_dtypes(include=np.number).columns
        df[num_cols] = scaler.fit_transform(df[num_cols])

    st.write("Cleaned Dataset:")
    st.write(df.head())

with tabs[2]:
    st.subheader("Exploratory Data Analysis")
    st.write("Basic Info:")
    st.write(df.describe())

    st.write("Missing Values:")
    st.write(df.isnull().sum())

    st.write("Correlation Heatmap:")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

with tabs[3]:
    st.subheader("Data Visualization")
    col_to_plot = st.selectbox("Choose a column to visualize:", df.columns)

    col_type = "categorical" if df[col_to_plot].nunique() < 15 else "numerical"
    st.write(f"Auto-suggested visualization: {'Bar plot' if col_type == 'categorical' else 'Histogram'}")

    if col_type == "categorical":
        vc = df[col_to_plot].value_counts().reset_index()
        vc.columns = ['category', 'count']
        fig = px.bar(vc, x='category', y='count')
        st.plotly_chart(fig)
    else:
        fig = px.histogram(df, x=col_to_plot)
        st.plotly_chart(fig)

    if st.checkbox("Show Boxplot"):
        st.plotly_chart(px.box(df, y=col_to_plot))

    if st.checkbox("Show Scatter Matrix"):
        st.plotly_chart(px.scatter_matrix(df))

with tabs[4]:
    st.subheader("🤖 Train ML Model")

    target = st.selectbox("Select Target Variable", df.columns)
    features = [col for col in df.columns if col != target]
    X = df[features]
    y = df[target]

    problem_type = "classification" if y.nunique() < 15 and y.dtype in [int, np.int32, np.int64] else "regression"
    st.write(f"Auto-detected problem type: {problem_type.capitalize()}")
    override = st.selectbox("Override Problem Type", [problem_type, "classification", "regression"])

    model_type = override
    if model_type == "classification":
        model_choice = st.selectbox("Choose Model", ["Random Forest Classifier", "Logistic Regression"])
    else:
        model_choice = st.selectbox("Choose Model", ["Random Forest Regressor", "Linear Regression"])

    test_size = st.slider("Test Set Size", 0.1, 0.5, 0.3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    st.info("Training model...")
    with st.spinner("Please wait while training..."):
        time.sleep(1.5)
        if model_choice == "Random Forest Classifier":
            model = RandomForestClassifier()
        elif model_choice == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_choice == "Random Forest Regressor":
            model = RandomForestRegressor()
        else:
            model = LinearRegression()
        model.fit(X_train, y_train)

    st.success("Model trained!")
    y_pred = model.predict(X_test)

    if model_type == "classification":
        st.write("Accuracy:", accuracy_score(y_test, y_pred))
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))
    else:
        st.write("MAE:", mean_absolute_error(y_test, y_pred))
        st.write("MSE:", mean_squared_error(y_test, y_pred))
        st.write("R² Score:", r2_score(y_test, y_pred))

    df_out = X_test.copy()
    df_out["Actual"] = y_test.values
    df_out["Predicted"] = y_pred
    st.download_button("Download Predictions", df_out.to_csv(index=False), file_name="predictions.csv")

with tabs[5]:
    st.subheader("📌 Model Insights")

    if hasattr(model, "feature_importances_"):
        st.write("Feature Importances:")
        importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
        st.bar_chart(importances)
    elif hasattr(model, "coef_"):
        st.write("Feature Coefficients:")
        importances = pd.Series(model.coef_[0], index=features).sort_values(ascending=False)
        st.bar_chart(importances)
    else:
        st.info("Feature importance not available for this model.")

    if st.checkbox("Show Permutation Importance"):
        with st.spinner("Calculating permutation importance..."):
            result = permutation_importance(model, X_test, y_test, n_repeats=10)
            sorted_idx = result.importances_mean.argsort()
            perm_imp = pd.Series(result.importances_mean[sorted_idx], index=X_test.columns[sorted_idx])
            st.bar_chart(perm_imp)

st.caption("VK")
