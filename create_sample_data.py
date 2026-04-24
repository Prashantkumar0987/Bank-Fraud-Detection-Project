import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create sample fraud detection dataset
np.random.seed(42)
n_rows = 1000

# Generate random dates
start_date = datetime(2023, 1, 1)
dates = [start_date + timedelta(days=int(x)) for x in np.random.rand(n_rows) * 365]

# Create sample data
data = {
    'Unnamed: 0': range(n_rows),
    'trans_date_trans_time': dates,
    'cc_num': np.random.randint(4000000000000000, 4999999999999999, n_rows, dtype=np.int64),
    'merchant': np.random.choice(['Amazon', 'Walmart', 'Target', 'CVS', 'Starbucks'], n_rows),
    'category': np.random.choice(['online', 'grocery', 'gas', 'shopping'], n_rows),
    'amt': np.random.uniform(5, 500, n_rows),
    'first': np.random.choice(['John', 'Jane', 'Mike', 'Sarah'], n_rows),
    'last': np.random.choice(['Smith', 'Johnson', 'Williams', 'Brown'], n_rows),
    'street': np.random.choice(['Main St', 'Oak Ave', 'Pine Rd'], n_rows),
    'city': np.random.choice(['New York', 'Los Angeles', 'Chicago'], n_rows),
    'state': np.random.choice(['NY', 'CA', 'IL'], n_rows),
    'zip': np.random.randint(10000, 99999, n_rows),
    'dob': [start_date - timedelta(days=int(x)) for x in np.random.rand(n_rows) * 365 * 30],
    'trans_num': [f'TXN{i:06d}' for i in range(n_rows)],
    'merch_lat': np.random.uniform(-90, 90, n_rows),
    'merch_long': np.random.uniform(-180, 180, n_rows),
    'lat': np.random.uniform(-90, 90, n_rows),
    'long': np.random.uniform(-180, 180, n_rows),
    'city_pop': np.random.randint(10000, 10000000, n_rows),
    'job': np.random.choice(['Engineer', 'Teacher', 'Doctor', 'Manager'], n_rows),
    'dob_year': np.random.randint(1950, 2000, n_rows),
    'is_fraud': np.random.choice([0, 1], n_rows, p=[0.99, 0.01])  # 1% fraud rate
}

df = pd.DataFrame(data)
df.to_csv('fraudTrain.csv', index=False)
print("Sample fraudTrain.csv created successfully!")
print(f"Created {len(df)} rows with {len(df.columns)} columns")
