# functions/single_positive_metrics_from_ranks.py

def single_positive_metrics_from_ranks(ranks, ks=(1,5,10,20,50,100)):
    """
    For single-positive:
      Recall@k = 1 if rank<=k else 0
      AP@100   = 1/rank if rank<=100 else 0  (== RR@100)
      MAP@100  = mean(AP@100) over queries
    Returns:
      recall_per_k: dict[k] -> dict[qid]->0/1
      ap100: dict[qid] -> float
    """
    recall_per_k = {k: {} for k in ks}
    ap100 = {}

    for qid, r in ranks.items():
        for k in ks:
            recall_per_k[k][qid] = 1.0 if (r is not None and r <= k) else 0.0

        ap100[qid] = (1.0 / r) if (r is not None and r <= 100) else 0.0

    return recall_per_k, ap100