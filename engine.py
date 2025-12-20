import os
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from insights import build_features_for_model

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

ESSENTIAL_CATEGORIES = ["utilities", "internet", "mobile"]
OPTIONAL_CATEGORIES = ["entertainment", "music", "cloud", "shopping"]

def load_data(file):
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df['Type'].str.lower() == 'debit']
    df['Transaction Description'] = df['Transaction Description'].str.lower().str.strip()
    df['Category'] = df['Category'].str.lower().str.strip()
    return df


def classify_necessity(category):
    if category in ESSENTIAL_CATEGORIES:
        return "Essential"
    elif category in OPTIONAL_CATEGORIES:
        return "Optional"
    else:
        return "Luxury"


def detect_subscriptions(df):
    results = []

    for desc, group in df.groupby('Transaction Description'):
        group = group.sort_values('Date')
        if len(group) < 3:
            continue

        amounts = group['Amount']
        dates = group['Date']
        mean_amt = amounts.mean()

        amount_consistency = np.mean(
            [abs(a - mean_amt) / mean_amt <= 0.05 for a in amounts]
        )

        gaps = dates.diff().dt.days.dropna()
        frequency_score = np.mean([(25 <= g <= 35) for g in gaps]) if not gaps.empty else 0

        months_active = dates.dt.to_period("M").nunique()

        confidence = (
            0.4 * amount_consistency +
            0.4 * frequency_score +
            0.2 * min(months_active / 6, 1)
        ) * 100

        if confidence < 60:
            continue

        category = group['Category'].iloc[0]
        necessity = classify_necessity(category)

        forgetfulness_factor = 1 if mean_amt < 500 else 0.7
        lsi = mean_amt * months_active * forgetfulness_factor

        results.append({
            "Subscription": desc.title(),
            "Category": category.title(),
            "Necessity": necessity,
            "Monthly Cost (₹)": round(mean_amt, 2),
            "Months Active": months_active,
            "Confidence (%)": round(confidence, 1),
            "Leakage Severity Index": round(lsi, 2),
            "Annual Savings if Cancelled (₹)": round(mean_amt * 12, 2),
            "5Y Savings if Cancelled (₹)": round(mean_amt * 12 * 5, 2)
        })

    return pd.DataFrame(results)


def detect_redundancy(subs):
    redundant = []
    for category, group in subs.groupby("Category"):
        if len(group) > 1:
            redundant.append({
                "Category": category,
                "Services": list(group["Subscription"]),
                "Potential Monthly Saving (₹)": round(group["Monthly Cost (₹)"].min(), 2)
            })
    return pd.DataFrame(redundant)


def user_risk_profile(subs):
    if subs.empty:
        return {}

    optional_ratio = subs[subs["Necessity"] == "Optional"]["Monthly Cost (₹)"].sum() / subs["Monthly Cost (₹)"].sum()

    return {
        "Total Subscriptions": len(subs),
        "Optional Leakage Ratio": round(optional_ratio * 100, 1),
        "Monthly Fixed Burn (₹)": round(subs["Monthly Cost (₹)"].sum(), 2),
        "Risk Level": "High" if optional_ratio > 0.5 else "Moderate"
    }

def train_clustering(subs, n_clusters=5, save_path=None):
    X = build_features_for_model(subs)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    k = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    k.fit(Xs)
    mdl = {'kmeans': k, 'scaler': scaler, 'features': list(X.columns)}
    path = save_path or os.path.join(MODEL_DIR, 'kmeans.joblib')
    joblib.dump(mdl, path)
    return path

def train_churn_model(subs, target_col='Churn', save_path=None):
    if target_col not in subs.columns:
        raise ValueError(f"Target column '{target_col}' not in dataframe")
    X = build_features_for_model(subs)
    y = subs[target_col].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True)
    scaler = StandardScaler()
    scaler.fit(X)
    mdl = {'model': rf, 'scaler': scaler, 'features': list(X.columns)}
    path = save_path or os.path.join(MODEL_DIR, 'churn_rf.joblib')
    joblib.dump(mdl, path)
    return path, report

def predict(subs, kmeans_path=None, churn_path=None):
    results = subs.copy()
    X = build_features_for_model(subs)
    if kmeans_path and os.path.exists(kmeans_path):
        kobj = joblib.load(kmeans_path)
        Xs = kobj['scaler'].transform(X[kobj['features']])
        results['cluster'] = kobj['kmeans'].predict(Xs)
    if churn_path and os.path.exists(churn_path):
        cobj = joblib.load(churn_path)
        Xc = cobj['scaler'].transform(X[cobj['features']])
        results['churn_prob'] = cobj['model'].predict_proba(Xc)[:,1]
        results['churn_pred'] = (results['churn_prob'] > 0.5).astype(int)
    # add heuristic scores and recommendations
    from insights import compute_risk_score, recommend_action, detect_anomalies
    results['risk_score'] = results.apply(compute_risk_score, axis=1)
    results['recommendation'] = results.apply(recommend_action, axis=1)
    
    # safely merge anomalies (only if Subscription column exists)
    if 'Subscription' in results.columns:
        anomalies = detect_anomalies(subs)
        if not anomalies.empty and 'Subscription' in anomalies.columns:
            results = results.merge(anomalies[['Subscription','is_anomaly','z_score']], on='Subscription', how='left')
    else:
        results['is_anomaly'] = False
        results['z_score'] = 0.0
    
    return results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-clusters', help='CSV file to train clustering')
    parser.add_argument('--train-churn', help='CSV file to train churn model (needs Churn column)')
    args = parser.parse_args()
    if args.train_clusters:
        df = pd.read_csv(args.train_clusters)
        path = train_clustering(df)
        print("Saved clustering model to", path)
    if args.train_churn:
        df = pd.read_csv(args.train_churn)
        path, rpt = train_churn_model(df)
        print("Saved churn model to", path)
        print("Churn report keys:", list(rpt.keys()))
