# functions/promptagator_filter.py

import os
import torch
from datasets import load_dataset

@torch.inference_mode()  # Inference-/Eval-Modus, keine Gradienten nötig
def filter_ai_dataset_topk(st_model, corpus, dataset_location, max_required_rank, device, asymmetric_encoder,
                           backbone_name, doc_batchsize, q_batchsize):

    # corpus: {chunk_id: passage_text}
    doc_ids = list(corpus.keys())
    doc_texts = [corpus[k] for k in doc_ids]
    doc_emb = st_model.encode(
        doc_texts,
        batch_size=doc_batchsize,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True
    ).to(device)

    docid2idx = {d: i for i, d in enumerate(doc_ids)}        # chunk_id -> row index in doc_emb

    unfiltered_ds = load_dataset(                            # Laden des vollständigen synthetischen Datensatzes
        "json",
        data_files=f"{dataset_location}/train_dataset.json",
        split="train"
    )

    qids, qtexts, qchunks = unfiltered_ds["id"], unfiltered_ds["anchor"], unfiltered_ds["chunk_id"]
    kept = set()                                            # Set aller zu behaltenden Fragen

    for s in range(0, len(qids), q_batchsize):              # gehe Queries in Batches durch (in Schritten von q_batchsize)
        b_qids   = qids[s:s + q_batchsize]                    # IDs dieses Query-Batches
        b_qtexts = qtexts[s:s + q_batchsize]                  # Texte (Fragen) dieses Batches
        b_chunks = qchunks[s:s + q_batchsize]                 # Gold chunk_id pro Query (entspricht corpus-key)

        q_emb = st_model.encode(                              # Query-Embeddings für den ganzen Batch berechnen
            b_qtexts,
            batch_size=q_batchsize,
            convert_to_tensor=True,
            normalize_embeddings=True,                        # L2-Norm → Cosine = Dot Product
            show_progress_bar=False
        ).to(device)

        # Scores aller Queries gegen alle Dokumente: [Batch, Dim] @ [Dim, Docs] -> [Batch, Docs]
        # Danach pro Query die Top-k Dokument-Indizes (Row-Indizes in doc_emb) holen
        topk_idx = torch.topk(q_emb @ doc_emb.T, k=max_required_rank, dim=1).indices.cpu().tolist()
        
        for i, qid in enumerate(b_qids):                    # jede Query im Batch einzeln prüfen
            gold_chunk = b_chunks[i]                          # korrekter Chunk (Key im corpus)
            gold_idx = docid2idx.get(gold_chunk)              # Index dieses Chunks in doc_emb (Zeilennummer)

            if gold_idx is not None and gold_idx in topk_idx[i]:
                kept.add(qid)

    # Behalte nur Datensatzeinträge, deren Query-ID im Set "kept" steht
    filtered_ds = unfiltered_ds.filter(lambda ex: ex["id"] in kept)

    if asymmetric_encoder:
        out_path = f"datasets/GerManualDPR/filtered_synthetic_asymmetric/{backbone_name}/train_dataset.json"
    else:
        out_path = f"datasets/GerManualDPR/filtered_synthetic/{backbone_name}/train_dataset.json"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    filtered_ds.to_json(out_path, orient="records", lines=False, force_ascii=False)

    print(f"kept={len(kept)}/{len(qids)} ({len(kept)/max(1,len(qids)):.2%}), saved: {out_path}")
    return