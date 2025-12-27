import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import os
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Configuration
LOCALE = 'pt_BR'
NUM_STORES = 100
NUM_PRODUCTS = 200
HISTORY_YEARS = 3
START_DATE = datetime.now() - timedelta(days=365 * HISTORY_YEARS)

# Init Faker
fake = Faker(LOCALE)
np.random.seed(42)  # Reproducibility

OUTPUT_DIR = "/opt/data/raw"

REGIONAL_GEO_MAP = {
    'Regional_Sul': {
        'RS': ['Porto Alegre', 'Caxias do Sul', 'Pelotas', 'Santa Maria'],
        'SC': ['Florianópolis', 'Joinville', 'Balneário Camboriú'],
        'PR': ['Curitiba', 'Londrina', 'Maringá']
    },
    'Regional_SP': {
        'SP': ['São Paulo', 'Campinas', 'Santos', 'Ribeirão Preto', 'Sorocaba']
    },
    'Regional_Rio_Minas': {
        'RJ': ['Rio de Janeiro', 'Niterói', 'Petrópolis'],
        'MG': ['Belo Horizonte', 'Uberlândia', 'Ouro Preto'],
        'ES': ['Vitória', 'Vila Velha']
    },
    'Regional_Nordeste': {
        'BA': ['Salvador', 'Feira de Santana'],
        'PE': ['Recife', 'Caruaru'],
        'CE': ['Fortaleza'],
        'RN': ['Natal']
    },
    'Regional_Centro_Norte': {
        'DF': ['Brasília'],
        'GO': ['Goiânia'],
        'AM': ['Manaus'],
        'PA': ['Belém']
    }
}

class DataGenerator:
    def __init__(self):
        self.store_ids = [fake.uuid4() for _ in range(NUM_STORES)]
        self.product_ids = [fake.uuid4() for _ in range(NUM_PRODUCTS)]
        self.product_prices = {}
        
    def generate_products(self):
        """Generates Product Dimension"""
        print("Generating Products...")
        categories = ['Camisetas', 'Calças', 'Vestidos', 'Acessórios', 'Casacos']
        
        data = []
        for pid in self.product_ids:
            cat = np.random.choice(categories)
            price_base = np.random.uniform(30, 200)
            cost_factor = np.random.uniform(0.2, 0.6)
            data.append({
                'product_id': pid,
                'name': f"{cat} {fake.color_name()} {fake.word()}",
                'category': cat,
                'base_price': round(price_base, 2),
                'cost': round(price_base * cost_factor, 2)
            })

            self.product_prices[pid] = round(price_base, 2)
        
        df = pd.DataFrame(data)
        df.to_parquet(f"{OUTPUT_DIR}/dim_product.parquet", index=False)

    def generate_stores(self):
        """Generates Physical and Management Store Dimensions with Geo-Coherence"""
        print("Generating Stores with Geographic Coherence...")
        
        placements = ['Shopping Mall', 'Street', 'Commercial Bldg', 'In-Company']
        
        # Cluster probabilities (Gold, Silver, Bronze)
        cluster_probs = [0.80, 0.15, 0.05]
        cluster_labels = ['C', 'B', 'A'] 
        
        physical_data = []
        mgmt_data = []
        
        regionals_list = list(REGIONAL_GEO_MAP.keys())
        
        for sid in self.store_ids:
            # 1. Geographic Hierarchy Selection (Top-Down)
            # Pick a regional
            regional = np.random.choice(regionals_list)
            
            # Pick a state within that regional
            available_states = list(REGIONAL_GEO_MAP[regional].keys())
            uf = np.random.choice(available_states)
            
            # Pick a city within that state
            city = np.random.choice(REGIONAL_GEO_MAP[regional][uf])
            
            # 2. Physical Characteristics
            placement = np.random.choice(placements, p=[0.6, 0.2, 0.1, 0.1])
            
            if placement == 'In-Company':
                size = np.random.normal(30, 5)
            elif placement == 'Shopping Mall':
                size = np.random.normal(150, 40)
            else:
                size = np.random.normal(80, 20)
            
            storage = size * np.random.uniform(0.1, 0.3)
            
            # Generate a realistic street address for that city
            street_name = fake.street_name()
            street_number = fake.building_number()
            full_address = f"{street_name}, {street_number} - {city} - {uf}"
            
            physical_data.append({
                'store_id': sid,
                'size_m2': round(size, 2),
                'placement': placement,
                'storage_capacity_m2': round(storage, 2),
                'address': full_address,
                'city': city,
                'state': uf  # Added State for easier analytics later
            })
            
            # 3. Management Characteristics
            cluster = np.random.choice(cluster_labels, p=cluster_probs)
            
            mgmt_data.append({
                'store_id': sid,
                'store_name': f"YouCom {city} {street_number} - {fake.word()}", # Name matches city
                'regional': regional,
                'current_cluster_label': cluster,
                'opening_date': fake.date_between(start_date='-5y', end_date='-1m')
            })
            
        # Save as Parquet
        pd.DataFrame(physical_data).to_parquet(f"{OUTPUT_DIR}/dim_store_physical.parquet", index=False)
        pd.DataFrame(mgmt_data).to_parquet(f"{OUTPUT_DIR}/dim_store_management.parquet", index=False)
        
        return pd.DataFrame(mgmt_data)

    def _sales_generator(self, store_df):
        """
        Generator Function: Yields monthly batches of sales data.
        This prevents OOM errors by not holding 3 years of data in RAM.
        """
        dates = pd.date_range(start=START_DATE, end=datetime.now(), freq='D')
        
        # Pre-compute store multipliers based on cluster (The 80/15/5 realization)
        # Cluster A (Gold) sells more than C (Bronze)
        store_multipliers = {
            sid: (np.random.normal(4.0, 6.0) if cl == 'A' else np.random.normal(1.0, 3.0) if cl == 'B' else np.random.normal(0.8, 1.2))
            for sid, cl in zip(store_df['store_id'], store_df['current_cluster_label'])
        }

        print(f"Streaming Sales generation for {len(dates)} days...")

        for date in dates:
            daily_sales = []
            
            # Seasonality: Chance to sell more on weekends (5=Sat, 6=Sun)
            weekday_factor = np.random.normal(1.0, 1.5) if date.dayofweek >= 5 else np.random.normal(1.0, 1.2)
            
            for sid in self.store_ids:
                base_volume = store_multipliers[sid] * weekday_factor
                
                # Randomly select products sold today (not every store sells every product every day)
                # Poisson distribution for number of transactions
                num_transactions = np.random.poisson(base_volume * 5) 
                
                if num_transactions == 0:
                    continue
                    
                selected_products = np.random.choice(self.product_ids, size=num_transactions)
                
                # Vectorized creation of the day's rows for this store
                for pid in selected_products:
                    qty = np.random.choice([1, 1, 1, 2, 3], p=[0.7, 0.15, 0.1, 0.04, 0.01])
                    base_price = self.product_prices[pid]
                    discount_rate = np.random.choice([0.0, 0.10, 0.20], p=[0.70, 0.20, 0.10])
                    final_price = base_price * (1 - discount_rate)
                    daily_sales.append({
                        'date': date.date(),
                        'store_id': sid,
                        'product_id': pid,
                        'qty': qty,
                        'unit_price_at_sale': round(final_price, 2)
                    })
            
            # Yield this day's dataframe
            if daily_sales:
                yield pd.DataFrame(daily_sales)

    def generate_sales(self, store_df):
        """Consumes the generator and writes to Parquet partitioned by Year/Month"""
        batch_buffer = []
        
        for daily_df in self._sales_generator(store_df):
            batch_buffer.append(daily_df)
            
            # Flush every 30 days to disk to keep RAM usage low
            if len(batch_buffer) >= 30:
                print(f"Flushing batch... {daily_df['date'].iloc[0]}")
                concat_df = pd.concat(batch_buffer)
                
                # PySpark loves partitionBy used here
                concat_df['year'] = pd.to_datetime(concat_df['date']).dt.year
                concat_df['month'] = pd.to_datetime(concat_df['date']).dt.month
                
                table = pa.Table.from_pandas(concat_df)
                pq.write_to_dataset(
                    table,
                    root_path=f"{OUTPUT_DIR}/fact_sales",
                    partition_cols=['year', 'month']
                )
                batch_buffer = []
        
        # Flush remaining
        if batch_buffer:
            concat_df = pd.concat(batch_buffer)
            concat_df['year'] = pd.to_datetime(concat_df['date']).dt.year
            concat_df['month'] = pd.to_datetime(concat_df['date']).dt.month
            table = pa.Table.from_pandas(concat_df)
            pq.write_to_dataset(table, root_path=f"{OUTPUT_DIR}/fact_sales", partition_cols=['year', 'month'])

    def generate_inventory(self):
        """Generates Weekly inventory snapshots"""
        print("Generating Inventory (Weekly snapshots)...")
        # Simplified: Just random inventory levels loosely correlated with store size
        # A real simulation would deduct sales from inventory, but that's complex to create cleanly.
        
        weeks = pd.date_range(start=START_DATE, end=datetime.now(), freq='W-MON')
        
        # To avoid massive file, we do this in loops too
        for date in weeks:
            inventory_data = []
            for sid in self.store_ids:
                # Random 50 products per store per week
                prods = np.random.choice(self.product_ids, 50)
                for pid in prods:
                    inventory_data.append({
                        'snapshot_date': date.date(),
                        'store_id': sid,
                        'product_id': pid,
                        'qty_on_hand': np.random.randint(0, 100)
                    })
            
            df = pd.DataFrame(inventory_data)
            df['year'] = pd.to_datetime(df['snapshot_date']).dt.year
            
            table = pa.Table.from_pandas(df)
            pq.write_to_dataset(
                table,
                root_path=f"{OUTPUT_DIR}/fact_inventory",
                partition_cols=['year']
            )

    def generate_history(self, store_df):
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
        weeks = pd.date_range(start=START_DATE, end=datetime.now(), freq='W-MON')
        
        sales_buffer = []
        inventory_buffer = []
        
        for week_start in tqdm(weeks):
            week_end = week_start + timedelta(days=6)
            days_in_week = pd.date_range(week_start, week_end)
            
            for sid in self.store_ids:
                # 1. DEFINE WEEKLY ASSORTMENT (The "Items Sold")
                # Each store sells a random subset (e.g., 40 items) this week
                num_active = np.random.randint(20, 60)
                active_pids = np.random.choice(self.product_ids, num_active, replace=False)
                
                # 2. GENERATE INVENTORY (Snapshot of these items)
                for pid in active_pids:
                    inventory_buffer.append({
                        'snapshot_date': week_start.date(),
                        'store_id': sid,
                        'product_id': pid,
                        'qty_on_hand': np.random.randint(5, 50) # Assuming positive stock for active items
                    })

                # 3. GENERATE SALES (Daily transactions for these items)
                base_volume = store_multipliers[sid]
                
                for day in days_in_week:
                    # Weekend multiplier
                    if day.dayofweek >= 5: # Weekend
                        factor = np.random.normal(1.5, 0.3)
                    else:
                        factor = np.random.normal(1.0, 0.2)

                    factor = max(0.1, factor) # Safety Clamp
                    vol = base_volume * factor

                    # Poisson transaction count
                    num_txns = np.random.poisson(vol)
                    if num_txns == 0:
                        continue
                    
                    # Only pick from ACTIVE items
                    sold_pids = np.random.choice(active_pids, size=num_txns)
                    
                    for pid in sold_pids:
                        qty = np.random.choice([1, 2, 3], p=[0.9, 0.08, 0.02])
                        base_price = self.product_prices[pid]
                        discount = np.random.choice([0.0, 0.1, 0.2], p=[0.7, 0.2, 0.1])
                        
                        sales_buffer.append({
                            'date': day.date(),
                            'store_id': sid,
                            'product_id': pid,
                            'qty': qty,
                            'unit_price_at_sale': round(base_price * (1 - discount), 2)
                        })
            
            # --- BUFFER FLUSH LOGIC ---
            # Flush Sales Monthly
            if len(sales_buffer) > 50_000: 
                self._flush_sales(sales_buffer)
                sales_buffer = []
                
            # Flush Inventory Monthly (Inventory grows fast)
            if len(inventory_buffer) > 50_000:
                self._flush_inventory(inventory_buffer)
                inventory_buffer = []

        # Final Flush
        if sales_buffer:
            self._flush_sales(sales_buffer)
        if inventory_buffer:
            self._flush_inventory(inventory_buffer)

    def _flush_sales(self, data):
        df = pd.DataFrame(data)
        if df.empty:
            return
        df['year'] = pd.to_datetime(df['date']).dt.year
        df['month'] = pd.to_datetime(df['date']).dt.month
        table = pa.Table.from_pandas(df)
        pq.write_to_dataset(table, root_path=f"{OUTPUT_DIR}/fact_sales", partition_cols=['year', 'month'])

    def _flush_inventory(self, data):
        df = pd.DataFrame(data)
        if df.empty:
            return
        df['year'] = pd.to_datetime(df['snapshot_date']).dt.year
        table = pa.Table.from_pandas(df)
        pq.write_to_dataset(table, root_path=f"{OUTPUT_DIR}/fact_inventory", partition_cols=['year'])

if __name__ == "__main__":
    os.makedirs(f"{OUTPUT_DIR}/fact_sales", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/fact_inventory", exist_ok=True)
    
    gen = DataGenerator()
    gen.generate_products()
    store_df = gen.generate_stores()
    gen.generate_history(store_df)
    print("Generation Complete.")