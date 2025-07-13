# Hotel Review Sentiment Classifier

This project is a **BERT-based PyTorch text classifier** that predicts the sentiment of hotel reviews (positive or negative). It uses the Hugging Face Transformers library to fine-tune a pre-trained BERT model on a labeled dataset of 20,000 TripAdvisor hotel reviews from Kaggle.

---

## Problem

Classify hotel reviews as:

- **Positive** (ratings 4 or 5 stars)
- **Negative** (ratings 1 or 2 stars)

3-star "neutral" reviews are removed to make the task a clear binary classification.

---

## Dataset

- **Source**: [TripAdvisor Hotel Reviews (Kaggle)](https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews/data)
- 20,000 English-language hotel reviews with 1–5 star ratings.
- Preprocessing includes:
  - Removing 3-star neutral reviews
  - Mapping ratings to binary labels
  - Cleaning/lowercasing text
  - Splitting into train/validation CSVs with stratification

---

## Model

- **Architecture**: BERT (bert-base-uncased)
- **Library**: Hugging Face Transformers (PyTorch)
- Fine-tuned on the TripAdvisor dataset for binary classification.
- Tokenized with max sequence length of 128.

---

## Performance

- **Best Validation Accuracy**: ~96.9%
- Computed during training using Hugging Face Trainer with accuracy as the selection metric.
- Metric calculated on held-out validation set stratified from original data.

> *Achieved ~96.9% validation accuracy classifying hotel reviews as positive or negative.*

---

## Features

- Data preprocessing pipeline in Python
- Fine-tuning script using Hugging Face Trainer
- Training logs and checkpoints saved
- CLI script for interactive predictions

