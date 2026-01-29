# Financial Sentiment: Accuracy–Latency Trade-off (Baselines + RAG-KNN)

This project compares transformer models on **Financial PhraseBank (75Agree)** and measures:
- Accuracy
- Macro F1
- Inference latency (ms/sample)

It also includes a **RAG-style KNN retrieval** experiment (retrieve similar training sentences and append as context) to test impact on performance and latency.

## Run in Google Colab (Recommended)

Cell 1 — Clone repository
```python
!git clone https://github.com/mohammednaveel03/financial-sentiment-tradeoff.git
%cd financial-sentiment-tradeoff
```
Cell 2 — Install dependencies
```python
!pip -q install -r requirements.txt
```
Cell 3 — Run experiments (baseline + RAG)
```python
!python -m src.run --output_dir outputs --epochs 2 --batch_size 16 --rag_k 3
```

Outputs

After running, you will get:
outputs/results.csv
outputs/*baseline.json
outputs/*ragk3.json
outputs/tradeoff_baseline.png
outputs/tradeoff_rag.png


