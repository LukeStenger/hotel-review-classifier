from datasets import load_dataset, DatasetDict

def load_and_tokenize_train_val(
    train_csv_path,
    val_csv_path,
    tokenizer,
    text_column="Review",
    label_column="label",
    max_length=128
) -> DatasetDict:
    """
    Loads pre-split train and val CSVs, tokenizes them with BERT tokenizer,
    and returns a PyTorch-ready DatasetDict.
    """

    # Load both CSVs
    data_files = {
        "train": train_csv_path,
        "validation": val_csv_path
    }
    dataset: DatasetDict = load_dataset("csv", data_files=data_files)

    # Tokenization function
    def tokenize_fn(example):
        return tokenizer(
            example[text_column],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

    # Apply tokenizer to both splits
    tokenized_dataset = dataset.map(tokenize_fn, batched=True)

    # Rename label column
    tokenized_dataset = tokenized_dataset.rename_column(label_column, "labels")

    # Set PyTorch format
    tokenized_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    return tokenized_dataset
