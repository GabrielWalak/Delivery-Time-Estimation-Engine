# Isolation Forest for detecting delivery anomalies
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
        
def IsolationForestModel(df):
    # Learning based on V (volume) and W (weight)
    features = [
    'delivery_time_days',  # Key: process outcome
    'distance_km',         # Key: route difficulty
    'product_weight_g',    # Physics
    'product_vol_cm3',     # Physics
    'freight_value',
]

    X=df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    iso_forest = IsolationForest(contamination=0.01, random_state=42,n_estimators=100)
    iso_forest.fit(X_scaled)

    df['is_anomaly'] = iso_forest.predict(X_scaled) == -1

    # Display results
    num_anomalies = df['is_anomaly'].sum()
    print(f"Number of detected anomalies: {num_anomalies}")
    print(f"    Detected {num_anomalies} anomalies (bottlenecks) in dataset.")
    
    return df
