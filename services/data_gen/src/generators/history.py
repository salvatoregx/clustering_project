import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta
from tqdm import tqdm
from .. import config

def _flush_sales(data):
    df = pd.DataFrame(data)
    if df.empty:
        return
    df['year'] = pd.to_datetime(df['date']).dt.year
    df['month'] = pd.to_datetime(df['date']).dt.month
    table = pa.Table.from_pandas(df)
    pq.write_to_dataset(table, root_path=f"{config.OUTPUT_DIR}/fact_sales", partition_cols=['year', 'month'])

def _flush_inventory(data):
    df = pd.DataFrame(data)
    if df.empty:
        return
    df['year'] = pd.to_datetime(df['snapshot_date']).dt.year
    table = pa.Table.from_pandas(df)
    pq.write_to_dataset(table, root_path=f"{config.OUTPUT_DIR}/fact_inventory", partition_cols=['year'])

def generate(store_ids, product_ids, product_prices, product_costs, store_df):
    """
    Synchronized Generator:
    Iterates week-by-week to ensure Sales and Inventory match.
    """
    print("Generating Synchronized History (Sales + Inventory)...")
    
    # Pre-compute multipliers
    store_multipliers = {}
    for sid, cl in zip(store_df['store_id'], store_df['current_cluster_label']):
        if cl == 'A':
            val = np.random.normal(4.0, 1.0)
        elif cl == 'B':
            val = np.random.normal(2.0, 0.5)
        else:
            val = np.random.normal(1.0, 0.2)
        store_multipliers[sid] = max(0.1, val)

    # Time Iterator
    weeks = pd.date_range(start=config.START_DATE, end=datetime.now(), freq='W-MON')
    
    sales_buffer = []
    inventory_buffer = []
    
    for week_start in tqdm(weeks):
        week_end = week_start + timedelta(days=6)
        days_in_week = pd.date_range(week_start, week_end)
        
        for sid in store_ids:
            # 1. DEFINE WEEKLY ASSORTMENT
            num_active = np.random.randint(20, 60)
            active_pids = np.random.choice(product_ids, num_active, replace=False)
            
            # 2. GENERATE INVENTORY
            for pid in active_pids:
                inventory_buffer.append({
                    'snapshot_date': week_start.date(),
                    'store_id': sid,
                    'product_id': pid,
                    'qty_on_hand': np.random.randint(5, 50)
                })

            # 3. GENERATE SALES
            base_volume = store_multipliers[sid]
            
            for day in days_in_week:
                if day.dayofweek >= 5: # Weekend
                    factor = np.random.normal(1.5, 0.3)
                else:
                    factor = np.random.normal(1.0, 0.2)

                factor = max(0.1, factor)
                vol = base_volume * factor

                num_txns = np.random.poisson(vol)
                if num_txns == 0:
                    continue
                
                sold_pids = np.random.choice(active_pids, size=num_txns)
                
                for pid in sold_pids:
                    qty = np.random.choice([1, 2, 3], p=[0.9, 0.08, 0.02])
                    base_price = product_prices[pid]
                    discount = np.random.choice([0.0, 0.1, 0.2], p=[0.7, 0.2, 0.1])
                    final_unit_price = round(base_price * (1 - discount), 2)
                    cost = product_costs[pid]
                    profit = (final_unit_price * qty) - (cost * qty)
                    
                    sales_buffer.append({
                        'date': day.date(),
                        'store_id': sid,
                        'product_id': pid,
                        'qty': qty,
                        'unit_price_at_sale': final_unit_price,
                        'profit_at_sale': round(profit, 2)
                    })
        
        # --- BUFFER FLUSH LOGIC ---
        if len(sales_buffer) > 50_000: 
            _flush_sales(sales_buffer)
            sales_buffer = []
            
        if len(inventory_buffer) > 50_000:
            _flush_inventory(inventory_buffer)
            inventory_buffer = []

    # Final Flush
    if sales_buffer:
        _flush_sales(sales_buffer)
    if inventory_buffer:
        _flush_inventory(inventory_buffer)
