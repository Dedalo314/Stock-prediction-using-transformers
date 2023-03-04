import os
import requests
import time
from tqdm import tqdm

# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
APIKEY = os.environ.get("INVESTMENT_API_KEY", "")
SYMBOL = "TSLA"

for i in tqdm(range(1,3)):
    for j in tqdm(range(1,13), leave=False):
        CSV_URL = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol={SYMBOL}&interval=15min&slice=year{i}month{j}&apikey={APIKEY}'
        with requests.Session() as s:
            download = s.get(CSV_URL)
            decoded_content = download.content.decode('utf-8')
            with open(f"tsla_downloaded_year{i}month{j}.csv", "w") as downloaded_file:
                downloaded_file.write(decoded_content)
        time.sleep(21)
