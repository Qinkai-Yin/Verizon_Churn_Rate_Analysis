import streamlit as st
import pandas as pd
import joblib
import lightgbm as lgb

st.set_page_config(page_title="Verizon Default Predictor", layout="centered")
st.title("📉 Verizon Default Prediction App")
st.markdown("Upload a CSV file with customer info and get predicted default risk.")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("Verizon_lightgbm_model.pkl")

model = load_model()

# File uploader
uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    #st.subheader("🔍 Raw Preview")
    #st.dataframe(df.head())

    try:
        # Save original columns
        df_raw = df.copy()

        # Drop irrelevant columns and encode
        if "True default" in df.columns:
            df = df.drop(columns=["True default"])
        df = df.drop(columns=["year", "day","default"], errors="ignore")
        df_encoded = pd.get_dummies(df, columns=["gender", "pmttype"], drop_first=True)

        # Predict
        y_pred = model.predict(df_encoded)
        y_prob = model.predict_proba(df_encoded)[:, 1]

        # Append results
        df_raw["Predicted default"] = y_pred
        df_raw["Default probability"] = y_prob

        st.subheader("📊 Prediction Results")
        st.dataframe(df_raw)

        # Download link
        csv = df_raw.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Results as CSV", data=csv, file_name="predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
else:
    st.info("Awaiting file upload...")
