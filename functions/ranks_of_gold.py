# functions/ranks_of_gold.py

def ranks_of_gold(run, gold_map, k_max=100):
    """
    Returns: rank[qid] = 1..k_max wenn Gold in Top-k_max, sonst None
    """
    ranks = {}
    for qid, ranked_docids in run.items():
        g = gold_map.get(qid)
        if g is None:
            continue
        top = ranked_docids[:k_max]
        try:
            ranks[qid] = top.index(g) + 1  # 1-based
        except ValueError:
            ranks[qid] = None
    return ranks