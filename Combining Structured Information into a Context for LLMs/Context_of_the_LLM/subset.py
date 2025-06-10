import requests
import pandas as pd

# Request 100 rows from the Hugging Face dataset server
url = "https://datasets-server.huggingface.co/rows"
params = {
    "dataset": "princeton-nlp/SWE-bench_Verified",
    "config": "default",
    "split": "test",
    "offset": 0,
    "length": 100 
}

response = requests.get(url, params=params)
data = response.json()

# Extract rows into a DataFrame
rows = [row["row"] for row in data["rows"]]
df = pd.DataFrame(rows)

# Filter for the specific repository
df_astropy = df[df["repo"] == "astropy/astropy"]

# Save to a CSV or JSON file
df_astropy.to_csv("astropy_subset.csv", index=False)
