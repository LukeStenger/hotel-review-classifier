from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

MODEL_NAME = 'bert-base-uncased'
NUM_LABELS = 2
OUTPUT_DIR = './outputs'

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS
)

# Placeholder for data loading
def load_data():
    return None, None

train_data, val_data = load_data()

# MINIMAL, VERSION-SAFE TrainingArguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=8
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
)

if __name__ == "__main__":
    print("✅ Trainer is ready to go!")
    print(trainer)
