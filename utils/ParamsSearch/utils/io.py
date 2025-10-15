import json
import os

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
