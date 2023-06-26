import json
import requests


REQ_FILE = '/workspace/corise-mlops/week3/project/data/requests.json'
HOST = 'http://127.0.0.1:80'

def execute_requests():
    read_lines = []
    with open(REQ_FILE, 'r') as f:
        for line in f:
            read_lines.append(json.loads(line.strip()))

    for data in read_lines:    
        r = requests.post(f"{HOST}/predict", data=json.dumps(data))

        print(f"Response is {r.text}")


if __name__ == "__main__":
    execute_requests()