# functions/load_topk_docids_from_predictions_jsonl.py

import json

def load_topk_docids_from_predictions_jsonl(path: str):
    """
    Returns:
      run[qid] = [corpus_id1, corpus_id2, ...] in Rank-Reihenfolge
    """
    run = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = str(obj["query_id"])
            run[qid] = [str(r["corpus_id"]) for r in obj["results"]]
    return run