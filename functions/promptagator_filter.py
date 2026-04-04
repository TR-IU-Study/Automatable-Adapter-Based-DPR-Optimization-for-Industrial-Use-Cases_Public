# functions/promptagator_filter.py

import os
import json
import torch
from datasets import load_dataset

@torch.inference_mode()
def filter_ai_dataset_topk(
    st_model,
    corpus,
    dataset_location,
    max_required_rank,
    device,
    asymmetric_encoder,
    backbone_name,
    doc_batchsize,
    q_batchsize
):
    # corpus: {chunk_id: passage_text}
    # Typen vereinheitlichen, damit chunk_id-Matches sicher funktionieren
    doc_ids = [str(k) for k in corpus.keys()]
    doc_texts = [corpus[k] for k in corpus.keys()]

    doc_emb = st_model.encode(
        doc_texts,
        batch_size=doc_batchsize,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True
    ).to(device)

    docid2idx = {d: i for i, d in enumerate(doc_ids)}

    unfiltered_ds = load_dataset(
        "json",
        data_files=f"{dataset_location}/train_dataset.json",
        split="train"
    )

    qids = unfiltered_ds["id"]
    qtexts = unfiltered_ds["anchor"]
    qchunks = [str(x) for x in unfiltered_ds["chunk_id"]]

    kept = set()

    # Absicherung: topk darf nicht größer als Anzahl Dokumente sein
    k = min(max_required_rank, len(doc_ids))

    for s in range(0, len(qids), q_batchsize):
        b_qids = qids[s:s + q_batchsize]
        b_qtexts = qtexts[s:s + q_batchsize]
        b_chunks = qchunks[s:s + q_batchsize]

        q_emb = st_model.encode(
            b_qtexts,
            batch_size=q_batchsize,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False
        ).to(device)

        topk_idx = torch.topk(q_emb @ doc_emb.T, k=k, dim=1).indices.cpu().tolist()

        for i, qid in enumerate(b_qids):
            gold_idx = docid2idx.get(b_chunks[i])
            if gold_idx is not None and gold_idx in topk_idx[i]:
                kept.add(qid)

    filtered_ds = unfiltered_ds.filter(lambda ex: ex["id"] in kept)

    if asymmetric_encoder:
        out_path = f"datasets/GerManualDPR/filtered_synthetic_asymmetric/{backbone_name}/train_dataset.json"
    else:
        out_path = f"datasets/GerManualDPR/filtered_synthetic/{backbone_name}/train_dataset.json"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Explizit als EIN JSON-Array speichern
    filtered_records = filtered_ds.to_list()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(filtered_records, f, ensure_ascii=False, indent=2)

    print(f"kept={len(kept)}/{len(qids)} ({len(kept)/max(1, len(qids)):.2%}), saved: {out_path}")
