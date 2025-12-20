import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import tempfile
import os
from engine import load_data, detect_subscriptions, detect_redundancy, user_risk_profile, predict, train_clustering, train_churn_model, MODEL_DIR
from insights import generate_explanations, save_spend_plot

st.set_page_config("Subscription Leakage Intelligence", layout="wide")
st.title("Subscription Leakage Intelligence Platform")

file = st.file_uploader("Upload bank transaction CSV", type="csv")
if not file:
    st.stop()

df = load_data(file)

# DETECT subscriptions from raw transactions
st.write("**Detecting subscriptions from transactions...**")
subs = detect_subscriptions(df)

if subs.empty:
    st.warning("No subscriptions detected. Ensure CSV has 'Date', 'Transaction Description', 'Amount' columns.")
    st.stop()

st.write(f"**Found {len(subs)} unique subscriptions**")
st.dataframe(subs.head(), use_container_width=True)

redundant = detect_redundancy(subs)
profile = user_risk_profile(subs)

st.subheader("🧠 User Subscription Risk Profile")
c1, c2, c3 = st.columns(3)
c1.metric("Total Subscriptions", profile["Total Subscriptions"])
c2.metric("Optional Leakage (%)", profile["Optional Leakage Ratio"])
c3.metric("Risk Level", profile["Risk Level"])

# SUBSCRIPTIONS
st.subheader("🔍 Subscription Decision Intelligence")
st.dataframe(subs.sort_values("Leakage Severity Index", ascending=False), use_container_width=True)

# REDUNDANCY
if not redundant.empty:
    st.subheader("🔁 Redundant Subscriptions Detected")
    st.dataframe(redundant)

# SIMULATION
st.subheader("🧮 Cancel Impact Simulator")
if "Subscription" in subs.columns:
    choice = st.selectbox("Select subscription to simulate cancellation", subs["Subscription"])
    row = subs[subs["Subscription"] == choice].iloc[0]
    savings_annual = row.get('Annual Savings if Cancelled (₹)', 0)
    savings_5y = row.get('5Y Savings if Cancelled (₹)', 0)
    st.success(f"Canceling {choice} saves ₹{savings_annual:.0f} per year and ₹{savings_5y:.0f} in 5 years.")

# EXPLANATIONS
st.subheader("💡 Why were these flagged?")
for exp in generate_explanations(subs):
    st.info(exp)

# ANALYZE
if st.button("Analyze Subscriptions"):
    with st.spinner("Running analysis..."):
        kpath = os.path.join(MODEL_DIR, 'kmeans.joblib')
        cpath = os.path.join(MODEL_DIR, 'churn_rf.joblib')
        res = predict(subs, kpath if os.path.exists(kpath) else None, cpath if os.path.exists(cpath) else None)
        explanations = generate_explanations(res)
        tmp = tempfile.gettempdir()
        plot_path = os.path.join(tmp, 'spend_trend.png')
        save_spend_plot(df, plot_path)
        st.success(f"Analysis complete: {len(res)} rows processed")
        st.write("Explanations:", explanations[:5])
        st.image(plot_path)
