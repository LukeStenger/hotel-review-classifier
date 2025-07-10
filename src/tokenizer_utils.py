from transformers import BertTokenizer

def get_tokenizer():
    """
    Loads a pretrained BERT tokenizer.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer

def tokenize_batch(tokenizer, texts, max_length=128):
    """
    Tokenizes a batch of texts.
    """
    return tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
