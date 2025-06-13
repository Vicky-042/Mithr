import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

st.title("🔍 Simple AutoML Tool")
st.write("Upload your CSV, select a model, and see the magic!")

# Step 1: File Upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Data Preview")
    st.write(df.head())

    # Step 2: Select target variable
    target = st.selectbox("Select your target column", df.columns)

    # Step 3: Preprocess
    st.subheader("🧼 Data Cleaning")
    st.write("Dropping rows with missing values...")
    df = df.dropna()

    X = df.drop(columns=[target])
    y = df[target]

    # Encode categorical
    X = pd.get_dummies(X)
    if y.dtype == 'O':
        y = pd.factorize(y)[0]

    # Step 4: Model selection
    model_type = st.selectbox("Choose a model", ["Random Forest", "Logistic Regression"])

    # Step 5: Train/Test Split
    test_size = st.slider("Test size", 0.1, 0.5, 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Step 6: Train model
    if st.button("🚀 Train Model"):
        if model_type == "Random Forest":
            model = RandomForestClassifier()
        else:
            model = LogisticRegression(max_iter=1000)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Step 7: Evaluation
        st.subheader("✅ Accuracy")
        acc = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy: {acc:.2f}")

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

        # Step 8: Feature Importance
        if model_type == "Random Forest":
            st.subheader("📌 Feature Importance")
            importances = model.feature_importances_
            feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
            st.bar_chart(feat_imp.head(10))

        # Step 9: Download Predictions
        st.subheader("⬇️ Download Predictions")
        results = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
        st.download_button("Download CSV", results.to_csv(index=False), "predictions.csv", "text/csv")
