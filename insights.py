def generate_explanations(subs):
    explanations = []
    for idx, row in subs.iterrows():
        sub_name = row['Subscription'] if 'Subscription' in subs.columns else f"Sub_{idx}"
        necessity = row['Necessity'] if 'Necessity' in subs.columns else 'Unknown'
        months = int(row['Months Active']) if 'Months Active' in subs.columns else 0
        confidence = float(row['Confidence (%)']) if 'Confidence (%)' in subs.columns else 0
        savings = float(row['Annual Savings if Cancelled (₹)']) if 'Annual Savings if Cancelled (₹)' in subs.columns else 0
        
        explanations.append(
            f"{sub_name} is classified as {necessity} spending, "
            f"active for {months} months with {confidence:.0f}% "
            f"subscription confidence. Cancelling can save ₹{savings:.0f} annually."
        )
    return explanations

# additional features
import os
import numpy as np
import pandas as pd
from scipy.stats import zscore
import matplotlib.pyplot as plt

def compute_monthly_trend(subs, date_col='Last Payment Date', amount_col='Monthly Cost'):
    # try common column name variations
    date_candidates = [date_col, 'Date', 'Payment Date', 'Transaction Date', 'date']
    amount_candidates = [amount_col, 'Amount', 'Cost', 'Price', 'amount']
    
    actual_date_col = None
    actual_amount_col = None
    
    for col in date_candidates:
        if col in subs.columns:
            actual_date_col = col
            break
    
    for col in amount_candidates:
        if col in subs.columns:
            actual_amount_col = col
            break
    
    # if still missing, return empty series
    if not actual_date_col or not actual_amount_col:
        return pd.Series(dtype=float)
    
    df = subs.copy()
    df[actual_date_col] = pd.to_datetime(df[actual_date_col], errors='coerce')
    df = df.dropna(subset=[actual_date_col])
    
    if df.empty:
        return pd.Series(dtype=float)
    
    df['month'] = df[actual_date_col].dt.to_period('M')
    trend = df.groupby('month')[actual_amount_col].sum().sort_index()
    return trend

def detect_anomalies(subs, amount_col='Monthly Cost', z_thresh=3.0):
    if amount_col not in subs.columns:
        return pd.DataFrame(columns=['Subscription','is_anomaly','z_score'])
    vals = subs[amount_col].fillna(0).astype(float)
    zs = zscore(vals, nan_policy='omit')
    is_anom = np.abs(zs) > z_thresh
    
    # safely get Subscription column (default to index if missing)
    sub_col = subs['Subscription'] if 'Subscription' in subs.columns else subs.index
    
    return pd.DataFrame({
        'Subscription': sub_col,
        'is_anomaly': is_anom,
        'z_score': np.nan_to_num(zs)
    }).reset_index(drop=True)

def _necessity_score(cat):
    if pd.isna(cat): return 1.0
    cat = str(cat).lower()
    if 'essential' in cat: return 0.0
    if 'optional' in cat: return 0.5
    if 'luxury' in cat: return 1.0
    return 0.5

def compute_risk_score(row):
    # heuristic: higher = more recommend to cancel
    try:
        necessity = _necessity_score(row.get('Necessity'))
        confidence = 100 - float(row.get('Confidence (%)', 50))  # low confidence -> higher risk
        months = float(row.get('Months Active', 0))
        annual_saving = float(row.get('Annual Savings if Cancelled (₹)', 0))
        cost_weight = min(annual_saving / (months + 1), annual_saving) if annual_saving >=0 else 0
        score = (0.4 * necessity * 100) + (0.3 * confidence) + (0.2 * min(months, 60)) + (0.1 * min(cost_weight/1000, 100))
        return round(max(0, min(100, score)), 2)
    except Exception:
        return 50.0

def recommend_action(row):
    score = compute_risk_score(row)
    if score >= 75:
        return 'Cancel'
    if score >= 50:
        return 'Pause / Downgrade'
    return 'Keep'

def build_features_for_model(subs):
    df = subs.copy()
    n = len(df)

    def _get_numeric_series(col, default):
        if col in df.columns:
            return pd.to_numeric(df[col], errors='coerce').fillna(default)
        return pd.Series([default] * n, index=df.index)

    # numeric conversions with safe defaults (always Series)
    df['Months Active'] = _get_numeric_series('Months Active', 0)
    df['Confidence (%)'] = _get_numeric_series('Confidence (%)', 50)
    df['Annual Savings if Cancelled (₹)'] = _get_numeric_series('Annual Savings if Cancelled (₹)', 0)

    # categorical mapping for Necessity (ensure Series)
    if 'Necessity' in df.columns:
        df['Necessity_Score'] = df['Necessity'].apply(_necessity_score)
    else:
        df['Necessity_Score'] = pd.Series([_necessity_score(None)] * n, index=df.index)

    df['risk_score'] = df.apply(compute_risk_score, axis=1)

    features = df[['Months Active','Confidence (%)','Annual Savings if Cancelled (₹)','Necessity_Score','risk_score']].copy()
    features.columns = ['months_active','confidence_pct','annual_saving','necessity_score','risk_score']
    return features.fillna(0)

def save_spend_plot(subs, out_path):
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    trend = compute_monthly_trend(subs)
    plt.figure(figsize=(10, 5))
    
    if not trend.empty and len(trend) > 0:
        try:
            trend.index = trend.index.to_timestamp()
            plt.plot(trend.index, trend.values, marker='o', linewidth=2, markersize=6, color='#1f77b4')
            plt.title('Monthly Spend Trend', fontsize=14, fontweight='bold')
            plt.ylabel('Total Monthly Cost (₹)', fontsize=12)
            plt.xlabel('Month', fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
        except Exception as e:
            plt.text(0.5, 0.5, f'Plot error: {str(e)}', ha='center', va='center')
    else:
        plt.text(0.5, 0.5, 'No trend data available\n(Check date and amount columns)', 
                 ha='center', va='center', fontsize=12, wrap=True)
    
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()
