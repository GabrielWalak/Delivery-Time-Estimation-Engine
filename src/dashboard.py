import os
import streamlit as st
import httpx
import pandas as pd
import plotly.express as px
from src.loader import get_data
from src.processing import process_data
from src.prediction import train_and_evaluate
from src.model import IsolationForestModel


API_BASE_URL = os.environ.get("DELIVERY_API_URL", "http://localhost:8000")
PREDICTION_ENDPOINT = f"{API_BASE_URL}/predict"
PREDICTION_TIMEOUT = 10.0

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Logistics AI Control Tower", 
    layout="wide", 
    page_icon="🚚",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #ff4b4b;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        font-weight: bold;
        height: 3em;
    }
</style>
""", unsafe_allow_html=True)

def call_prediction_api(payload):
    """Call FastAPI prediction endpoint with httpx library."""
    try:
        response = httpx.post(PREDICTION_ENDPOINT, json=payload, timeout=PREDICTION_TIMEOUT)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        st.error(f"Prediction API error ({exc.response.status_code}): {exc.response.text}")
        return None
    except httpx.RequestError as exc:
        st.error(f"Prediction API unreachable: {exc}")
        return None
    return response.json()

# --- 2. SYSTEM LOADING (CACHE) ---
# This executes only once when the app starts!

@st.cache_resource
def load_system():
    with st.spinner('🚀 Starting AI engine...'):
        # 1. Data
        raw_data = get_data()
        df = process_data(raw_data)
        
        # 2. Anomalies
        df = IsolationForestModel(df)
        
        # 3. Training - Receive 5 elements (as returned by prediction.py)
        final_df, model, r2, mae, features = train_and_evaluate(df)
        
        # 4. Calculate biz_acc HERE (in dashboard)
        final_df['abs_error'] = abs(final_df['delivery_time_days'] - final_df['predicted_days'])
        biz_acc = (final_df['abs_error'] < 3).mean()
        
    # Return 6 things (5 from model + 1 calculated here)
    return final_df, model, r2, mae, features, biz_acc

# Start loading
try:
    df, model, r2, mae, features, biz_acc = load_system()
except Exception as e:
    st.error(f"⚠️ Critical error occurred: {e}")
    st.stop()

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("🚚 Olist Logistics")
    st.markdown("---")
    st.markdown("**Model Status:** 🟢 Ready")
    st.markdown(f"**Records:** {len(df):,}")
    st.markdown("---")
    st.info("""
    **About Project:**
    System predicts delivery time in Brazilian e-commerce using:
    - XGBoost (Regression)
    - Isolation Forest (Anomalies)
    - Haversine Distance
    - Infrastructure Analysis
    """)
    st.markdown("---")
    st.caption("Author: Junior Data Scientist")

# --- 4. MAIN VIEW - HEADER ---
st.title("AI Supply Chain Control Tower 🇧🇷")
st.markdown("### Predictive system and logistics anomaly detection")

# KPI Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Business Accuracy", f"{biz_acc:.1%}", "Error < 3 days")
col2.metric("R² Score", f"{r2:.1%}", "Model explainability")
col3.metric("Mean Error (MAE)", f"{mae:.2f} days", "Average error")
col4.metric("Detected Anomalies", f"{df['is_anomaly'].sum()}", "Bottlenecks")

st.divider()

# --- 5. APP TABS ---
tab_sim, tab_map, tab_xai = st.tabs(["🛠️ Delivery Simulator", "🗺️ Map & Anomalies", "🧠 Why? (XAI)"])

# === TAB 1: SIMULATOR ===
with tab_sim:
    st.subheader("Predict time for a new package")
    
    # Input container
    with st.container():
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown("##### 📦 Package Physics")
            weight = st.number_input("Weight (g)", 50, 30000, 1000, step=50)
            vol = st.number_input("Volume (cm³)", 100, 100000, 5000, step=100)
            freight = st.number_input("Shipping price (BRL)", 0, 1000, 20, step=5)
            
        with c2:
            st.markdown("##### 🌍 Route & Time")
            dist = st.slider("Distance (km)", 0, 4000, 500)
            month = st.select_slider("Purchase month", options=range(1, 13), value=11, help="11=November (Black Friday)")
            hour = st.slider("Purchase hour", 0, 23, 14)
            is_weekend = st.checkbox("Weekend purchase?")
            lag = st.number_input("Payment waiting days", 0, 10, 0)


    # Model limitations info
    st.info("""
    ℹ️ **Model Information:**
    Model trained on Olist data (2016-2018). Best accuracy for:
    - Distance: 50-2500 km
    - Weight: 200g - 15kg  
    - Standard months (not Black Friday)
    
    For extreme values or data outside this range, estimation may be less reliable.
    """)

    # Action button
    if st.button("🚀 CALCULATE ESTIMATE", type="primary"):
        payload = {
            'product_weight_g': weight,
            'product_vol_cm3': vol,
            'distance_km': dist,
            'customer_lat': df['customer_lat'].mean(),
            'customer_lng': df['customer_lng'].mean(),
            'seller_lat': df['seller_lat'].mean(),
            'seller_lng': df['seller_lng'].mean(),
            'payment_lag_days': lag,
            'is_weekend_order': bool(is_weekend),
            'freight_value': freight,
            'purchase_month': month,
        }

        api_response = call_prediction_api(payload)
        if api_response:
            pred_days = api_response.get('predicted_days')
            warnings = api_response.get('warnings', [])
            mae_value = api_response.get('mae')
            message = api_response.get('message')

            quality_note = f" · MAE ≃ {mae_value:.1f} days" if isinstance(mae_value, (int, float)) else ""
            st.success(f"📦 Estimated delivery time: **{pred_days:.1f} days**{quality_note}")

            if message:
                st.caption(message)

            if warnings:
                st.warning(f"""
                ⚠️ **Warning:** Input data contains extreme values: {', '.join(warnings)}.

                Model was trained mainly on 2017-2018 data.
                For extreme values, estimation may be less accurate.
                Error could be ±7-10 days instead of standard ±4 days.
                """)

            max_days = 30
            progress = float(min(pred_days / max_days, 1.0))

            if pred_days < 5:
                bar_color = "green"
                msg = "⚡ Fast delivery (Express)"
            elif pred_days < 15:
                bar_color = "orange"
                msg = "🚛 Standard time"
            else:
                bar_color = "red"
                msg = "⚠️ Long route / Delay"

            st.progress(progress, text=msg)


# === TAB 2: MAP AND ANOMALIES ===
with tab_map:
    col_m1, col_m2 = st.columns([3, 1])
    with col_m1:
        st.subheader("Geographic Distribution of Anomalies")
        # Filter only anomalies
        anomalies_df = df[df['is_anomaly'] == True]
        
        if not anomalies_df.empty:
            # Sample 2000 points so map doesn't lag
            st.map(
                anomalies_df.sample(min(2000, len(anomalies_df))),
                latitude='customer_lat',
                longitude='customer_lng',
                size=20,
                color='#ff0000'  # Red dots
            )
        else:
            st.warning("No anomalies to display.")
            
    with col_m2:
        st.markdown("### Statistics")
        st.write(f"Total anomalies: **{len(anomalies_df)}**")
        st.write("Anomalies are orders that are atypical (e.g., very long delivery time for short distance).")

# === TAB 3: XAI (FEATURE IMPORTANCE) ===
with tab_xai:
    st.subheader("What affects delivery time?")
    st.markdown("The chart below shows which factors are most important for the AI model.")
    
    # Extract feature importance from XGBoost model
    importances = model.feature_importances_
    # Get feature names
    feature_names = ['product_weight_g', 'product_vol_cm3', 'distance_km', 'customer_lat', 'customer_lng',
                'seller_lat', 'seller_lng', 'payment_lag_days', 'is_weekend_order', 'freight_value',
                'purchase_month']
    
    # Create DataFrame for chart
    feat_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=True)  # Sort ascending for horizontal chart
    
    # Plotly chart
    fig = px.bar(
        feat_df, 
        x='Importance', 
        y='Feature', 
        orientation='h',
        title="Impact of variables on delivery time (XGBoost Feature Importance)",
        color='Importance',
        color_continuous_scale='Blues'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("💡 **Conclusion:** Distance and Seasonality (Month) are key factors. Package physics has less impact.")