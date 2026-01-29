# Financial Sentiment Analysis  
## Accuracy–Latency Trade-off Using Transformer Models

This repository contains the implementation and experimental evaluation for a comparative study of transformer-based sentiment analysis models on the **Financial PhraseBank (75% agreement)** dataset.

The primary objective of this project is to analyze the **trade-off between predictive performance (Accuracy, Macro-F1)** and **inference latency**, which is a critical consideration for deploying NLP models in real-world financial systems.

In addition to baseline model evaluation, the project includes an optional **retrieval-augmented (RAG-style)** experiment, where nearest-neighbor financial sentences are appended as contextual evidence.

---

## Models Evaluated

The following pre-trained transformer models are evaluated:

- **FinBERT (ProsusAI/finbert)**  
  A BERT-based model fine-tuned on financial texts.

- **DistilBERT-SST2**  
  A lightweight distilled transformer optimized for faster inference.

- **RoBERTa-Twitter (cardiffnlp)**  
  A RoBERTa-based sentiment model with strong performance on short text.

---

## Dataset

- **Financial PhraseBank**
- Agreement level: **75%**
- Task: **Three-class sentiment classification**
  - Negative
  - Neutral
  - Positive

The dataset is automatically downloaded, extracted, and parsed during execution.  
No manual dataset setup is required.

---

## Repository Structure

financial-sentiment-tradeoff/
│
├── src/
│   ├── run.py        # Main experiment runner
│   ├── data.py       # Dataset download, parsing, and splitting
│   ├── utils.py      # Helper utilities
│   └── rag_knn.py    # Retrieval-augmented (RAG-style) logic
│
├── outputs/
│   ├── results.csv              # Quantitative experiment results
│   ├── performance_bar.png      # Accuracy & Macro-F1 comparison
│   ├── latency_bar.png          # Latency comparison
│   └── tradeoff_scatter.png     # Accuracy–Latency trade-off visualization
│
├── requirements.txt
└── README.md


---

## How to Run (Google Colab – Recommended)

Open a new Google Colab notebook and run the following commands **in separate cells**.

---

### Step 1 — Clone the repository

```python
!git clone https://github.com/mohammednaveel03/financial-sentiment-tradeoff.git
%cd financial-sentiment-tradeoff
```

### Step 2 — Install dependencies

```python
!pip install -r requirements.txt
```

### Step 3 — Run baseline experiments

```python
!python -m src.run --output_dir outputs --agree 75 --rag_k 0
```

### Step 4 — Run retrieval-augmented (RAG-KNN) experiments

```python
!python -m src.run --output_dir outputs --agree 75 --rag_k 3
```

### Step 5 — View outputs

```python
!ls -lah outputs
!cat outputs/results.csv
```

---

## Outputs

After execution, the following files are generated in the `outputs/` directory:

- **results.csv**  
  Accuracy, Macro-F1 score, and inference latency for each model.

- **performance_bar.png**  
  Visual comparison of Accuracy and Macro-F1.

- **latency_bar.png**  
  Inference latency comparison (milliseconds per sample).

- **tradeoff_scatter.png**  
  Accuracy–latency trade-off visualization.

---

## Key Observations

- Lightweight models (**DistilBERT**) achieve lower inference latency but reduced accuracy.
- Larger models (**RoBERTa**) achieve higher accuracy at the cost of increased latency.
- Domain-specific models (**FinBERT**) are not always optimal for short financial phrases.
- Retrieval-augmented context affects both performance and inference cost, highlighting real deployment trade-offs.

---

## Reproducibility

- All experiments are deterministic given the same environment.
- No external dataset scripts are required.
- The entire pipeline can be executed end-to-end using the instructions above.

---

## Author

**Mohammed Naveel**  
Master’s Student – Artificial Intelligence  
Bahçeşehir University
