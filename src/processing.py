import pandas as pd
import numpy as np

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates distance in kilometers between two points on Earth (Haversine Formula).
    """
    R = 6371  # Earth radius in kilometers

    # Convert degrees to radians
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    # Haversine formula
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c

def process_data(dfs):
    print(">>> [Processing] Starting feature engineering with Geolocation...")

    orders = dfs['orders']
    items = dfs['items']
    products = dfs['products']
    customers = dfs['customers']
    sellers = dfs['sellers']
    geo = dfs['locations']

    # STEP 1: Fix Geolocation (This is crucial!)
    # Geo table has duplicates. Group by zip code and take mean position.
    geo = geo.groupby('geolocation_zip_code_prefix').agg({
        'geolocation_lat': 'mean',
        'geolocation_lng': 'mean'
    }).reset_index()

    # STEP 2: Join Main Tables
    # Orders -> Items -> Products
    # Using inner join, as we only want orders with products
    main_df = orders.merge(items, on='order_id')
    main_df = main_df.merge(products, on='product_id')
    
    # Join Customer data (to get their zip code)
    main_df = main_df.merge(customers, on='customer_id')
    
    # Join Seller data (to get their zip code)
    main_df = main_df.merge(sellers, on='seller_id')

    # STEP 3: Join Coordinates (TWICE!)
    
    # A. Where is the CUSTOMER? (Join on customer_zip_code_prefix)
    main_df = main_df.merge(
        geo, 
        left_on='customer_zip_code_prefix', 
        right_on='geolocation_zip_code_prefix', 
        how='left'
    ).rename(columns={
        'geolocation_lat': 'customer_lat',
        'geolocation_lng': 'customer_lng'
    })

    # B. Where is the SELLER? (Join on seller_zip_code_prefix)
    main_df = main_df.merge(
        geo, 
        left_on='seller_zip_code_prefix', 
        right_on='geolocation_zip_code_prefix', 
        how='left',
        suffixes=('', '_seller')  # Important to avoid name conflicts
    ).rename(columns={
        'geolocation_lat': 'seller_lat',
        'geolocation_lng': 'seller_lng'
    })

    # STEP 4: Calculations and Physics
    
    # Volume
    main_df['product_vol_cm3'] = (
        main_df['product_length_cm'].fillna(0) * main_df['product_height_cm'].fillna(0) * main_df['product_width_cm'].fillna(0)
    )

    # STEP 4: Time and Process Calculations (Process Mining)
    
    # Convert key dates
    main_df['purchase_date'] = pd.to_datetime(main_df['order_purchase_timestamp'])
    main_df['approved_date'] = pd.to_datetime(main_df['order_approved_at'])
    main_df['delivered_date'] = pd.to_datetime(main_df['order_delivered_customer_date'])
    
    # A. Delivery Time (Target)
    main_df['delivery_time_days'] = (main_df['delivered_date'] - main_df['purchase_date']).dt.days

    # B. Payment Lag
    # How many days from "Click Buy" to "Payment Approved"?
    # Fill NaN with zeros (assume instant payment if no date)
    main_df['payment_lag_days'] = (main_df['approved_date'] - main_df['purchase_date']).dt.days.fillna(0)
    
    # C. Weekend Effect (Day of Week)
    # 0 = Monday, 6 = Sunday
    main_df['purchase_day_of_week'] = main_df['purchase_date'].dt.dayofweek
    main_df['is_weekend_order'] = (main_df['purchase_day_of_week'] >= 4).astype(int)
    
    # Purchase month
    main_df['purchase_month'] = main_df['purchase_date'].dt.month

    # STEP 5: DISTANCE CALCULATION (Haversine)
    # We have customer_lat/lng and seller_lat/lng. Calculate distance.
    main_df['distance_km'] = haversine_distance(
        main_df['customer_lat'], main_df['customer_lng'],
        main_df['seller_lat'], main_df['seller_lng']
    )

    # Cleanup (Remove empty rows, e.g., no delivery date)
    # Keep only delivered orders
    final_df = main_df[main_df['order_status'] == 'delivered'].dropna(subset=[
        'delivery_time_days', 'product_weight_g', 'distance_km'
    ])

    # Select columns for model
    cols_to_keep = [
        'order_id',
        'delivery_time_days',
        'product_weight_g',
        'product_vol_cm3',
        'distance_km',
        'freight_value',
        'payment_lag_days',
        'is_weekend_order',
        'customer_lat',
        'customer_lng',
        'seller_lat',
        'seller_lng',
        'purchase_month',
    ]
    
    final_df = final_df[cols_to_keep]

    print(f">>> [Processing] Done! Records: {len(final_df)}")
    return final_df