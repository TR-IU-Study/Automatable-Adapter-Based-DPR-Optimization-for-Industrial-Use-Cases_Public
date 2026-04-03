# functions/load_single_gold_map.py

import json

def load_single_gold_map(test_dataset_json_path: str):
    """
    Returns: gold[qid] = gold_docid (beides als str)
    """
    with open(test_dataset_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(item["id"]): str(item["chunk_id"]) for item in data}