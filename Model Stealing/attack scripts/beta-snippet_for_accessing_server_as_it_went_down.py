import time
import requests

TOKEN = "55172888"
PORT = "9306"
SEED = "74714394"
API_HOST = "http://34.122.51.94"
ONNX_PATH = "stolen_encoder.onnx"

for attempt in range(5):
    try:
        with open(ONNX_PATH, "rb") as f:
            files = {"file": f}
            headers = {"token": TOKEN, "seed": SEED}
            response = requests.post(f"{API_HOST}:9090/stealing", files=files, headers=headers)
            print("✅ Submission response:", response.json())
            break
    except requests.exceptions.RequestException as e:
        print(f"⏳ Attempt {attempt+1}/5 failed: {e}")
        time.sleep(60)  # wait before retrying
