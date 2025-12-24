import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

def train_and_evaluate(df):
    """
    Main function controlling prediction.
    Performs two steps:
    1. Evaluation (Check quality on data subset).
    2. Final Training (Train model on all data).
    """
    print(">>> [Prediction] Starting prediction module (XGBoost)...")

    # --- FEATURE CONFIGURATION ---
    # Columns the model will learn from.
    features = ['product_weight_g', 'product_vol_cm3', 'distance_km', 'customer_lat', 'customer_lng',
                'seller_lat', 'seller_lng', 'payment_lag_days', 'is_weekend_order', 'freight_value',
                'purchase_month']
    target = 'delivery_time_days'   

    # --- DATA PREPARATION ---
    # Train only on "healthy" data (without anomalies)
    df_clean = df[df['is_anomaly'] == False].copy()
    
    X = df_clean[features]
    y = df_clean[target]

    # ==========================================
    # STEP 1: EVALUATION (How much % can we get?)
    # ==========================================
    print("    Phase 1: Testing model (Train/Test Split)...")
    
    # Split: 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # XGBoost configuration for testing
    model_test = xgb.XGBRegressor(
        n_estimators=1000,      # High limit
        learning_rate=0.05,     # Learn slowly and accurately
        max_depth=6,            # Tree depth
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50  # Overfitting protection!
    )

    # Train with test set monitoring
    model_test.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # Test results
    best_rounds = model_test.best_iteration  # How many rounds were optimal?
    y_pred = model_test.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"    >>> Test results:")
    print(f"        R^2 Score: {r2:.4f} (Variance explained)")
    print(f"        Mean error: {mae:.2f} days")
    print(f"        Optimal number of trees: {best_rounds}")

    # ==========================================
    # STEP 2: FINAL TRAINING (On 100% data)
    # ==========================================
    print("    Phase 2: Training final model on full dataset...")

    final_model = xgb.XGBRegressor(
        n_estimators=best_rounds,  # Use number discovered in Phase 1
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )

    # Train on EVERYTHING (X, y), not just X_train
    final_model.fit(X, y)

    # ==========================================
    # STEP 3: PREDICTION FOR ENTIRE TABLE
    # ==========================================
    # Add predictions to original table
    df['predicted_days'] = final_model.predict(df[features])
    df['prediction_error'] = df['delivery_time_days'] - df['predicted_days']

    print(">>> [Prediction] Done. Model returned.")
    
    # Return table with results and the model itself (for later saving)
    return df, final_model, r2, mae, features