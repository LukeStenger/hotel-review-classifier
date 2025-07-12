from transformers import BertForSequenceClassification, BertTokenizer
import torch

MODEL_PATH = "./models/saved_model/bert-hotel-classifier"

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model.eval()

def predict_sentiment(text):
    inputs = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
    return "Positive" if prediction == 1 else "Negative"

if __name__ == "__main__":
    while True:
        text = input("\nEnter hotel review (or blank to quit): ")
        if not text.strip():
            break
        sentiment = predict_sentiment(text)
        print(f"Predicted Sentiment: {sentiment}")
