from transformers import BertTokenizer

def get_tokenizer():
    """
    Loads a pretrained BERT tokenizer.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer


