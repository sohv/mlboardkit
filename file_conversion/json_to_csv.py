import pandas as pd
import json

# Load JSON data
with open('data.json') as f:
    data = json.load(f)

# Convert to DataFrame and then to CSV
df = pd.json_normalize(data)
df.to_csv('output.csv', index=False)
