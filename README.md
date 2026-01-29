Financial Sentiment Analysis
Accuracy–Latency Trade-off with Transformer Models

This repository contains the code and experimental results for a comparative study of transformer-based sentiment analysis models on the Financial PhraseBank (75% agreement) dataset.

The project focuses on analyzing the trade-off between predictive performance (accuracy, macro-F1) and inference latency, which is a key consideration for real-world deployment of NLP models in financial applications.

In addition to baseline evaluation, the project includes an optional retrieval-augmented (RAG-style) experiment, where nearest-neighbor financial sentences are appended as additional context.

Models Evaluated

The following pre-trained transformer models are evaluated:

FinBERT (ProsusAI/finbert)
Domain-specific BERT model trained on financial text.

DistilBERT-SST2
Lightweight distilled model optimized for faster inference.

RoBERTa-Twitter (cardiffnlp)
Robust RoBERTa-based sentiment model with strong generalization on short texts.

Dataset

Financial PhraseBank

Agreement level: 75%

Task: 3-class sentiment classification

Negative

Neutral

Positive

The dataset is automatically downloaded and parsed during execution.
No manual dataset setup is required.
