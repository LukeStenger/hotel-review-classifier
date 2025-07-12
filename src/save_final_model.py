from transformers import BertForSequenceClassification, BertTokenizer

CHECKPOINT_PATH = "./outputs/checkpoint-5493"
SAVE_PATH = "./models/saved_model/bert-hotel-classifier"

# Load model from best checkpoint
model = BertForSequenceClassification.from_pretrained(CHECKPOINT_PATH)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Save model and tokenizer
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

print(f"✅ Model and tokenizer saved to {SAVE_PATH}")
