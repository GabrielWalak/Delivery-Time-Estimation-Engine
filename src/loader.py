import pandas as pd
import kagglehub
import os

def get_data():
    """
    Downloads dataset path and then loads files manually.
    """
    print(">>> [Loader] Downloading dataset (files only)...")
    
    # 1. Get ONLY the folder path (without trying to load into table)
    path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")
    
    print(f">>> Dataset located at: {path}")
    

    dfs={}
    files_to_load = {"orders" : "olist_orders_dataset.csv",
                    "items" : "olist_order_items_dataset.csv",
                    "products" : "olist_products_dataset.csv",
                    "customers" : "olist_customers_dataset.csv",
                    "sellers" : "olist_sellers_dataset.csv",
                    "locations" : "olist_geolocation_dataset.csv",
                    }
    for key, file_name in files_to_load.items():
        csv_path = os.path.join(path, file_name)
        print(f">>> Loading file: {csv_path}")
        try:
            dfs[key] = pd.read_csv(csv_path)
        except UnicodeDecodeError:
            print(f">>> UTF-8 encoding error in {file_name}. Trying 'latin-1'...")
            dfs[key] = pd.read_csv(csv_path, encoding="latin-1")
    
    return dfs