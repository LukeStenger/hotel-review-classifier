from transformers import (
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from tokenizer_utils import get_tokenizer
from dataset import load_and_tokenize_train_val

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

tokenizer = get_tokenizer()

dataset = load_and_tokenize_train_val(
    train_csv_path='data/processed/train.csv',
    val_csv_path='data/processed/val.csv',
    tokenizer=tokenizer
)

train_dataset = dataset["train"]
eval_dataset = dataset["validation"]

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)

training_args = TrainingArguments(
    output_dir='./outputs',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./outputs/logs',
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)
 
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

trainer.train()

