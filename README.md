This project trains a BERT-based text classifier to predict the sentiment of hotel reviews.

It uses Hugging Face Transformers on top of PyTorch to:

Load and clean review data

Tokenize with a pretrained BERT tokenizer

Fine-tune BERT for binary sentiment classification

Evaluate using accuracy and F1 score