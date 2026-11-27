# Retrieval Ablation Study (commit `dd84e0f`)

| Cấu hình thực nghiệm | Recall@5 | MRR@5 | nDCG@5 | Độ trễ TB (ms) |
| :--- | :---: | :---: | :---: | :---: |
| 1. Dense Only (bge-small-en-v1.5) | 54.29% | 0.4357 | 0.4945 | 342.6 |
| 2. Sparse Only (Qdrant/bm25) | 70.00% | 0.6881 | 0.8265 | 239.1 |
| 3. Hybrid + RRF (k=60) | 68.57% | 0.5376 | 0.6686 | 208.7 |
| 4. Hybrid + RRF + ONNX Reranker (Laptop) | 65.71% | 0.5724 | 0.7628 | 2491.2 |
| 5. Hybrid + RRF + BGE Reranker (Kaggle GPU) | 72.86% | 0.6995 | 0.8241 | 627.0 |
