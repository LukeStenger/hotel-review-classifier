import os
import pandas as pd
from sklearn.model_selection import train_test_split


# Load raw CSV
def load_data(file_path):
    return pd.read_csv(file_path)

# Remove 3-star reviews
def remove_neutral_reviews(df):
    return df[df['Rating'] != 3]
   
# Map ratings to binary labels
def map_ratings_to_binary_labels(df):
    df['label'] = df['Rating'].apply(lambda r: 1 if r in [4,5] else 0)
    return df

# Clean the Review text column by lowercasing and stripping whitespace
def clean_text(df):
    df['Review'] = df['Review'].str.lower().str.strip()
    return df


# Drop rows with missing or empty text
def drop_empty_reviews(df):
    df = df[df['Review'].notnull()]
    df = df[df['Review'].str.strip() != '']
    return df

# Split into train/val
def split_train_val(df, test_size=0.2, random_state=42):
    """
    Splits the DataFrame into train and validation sets with stratification.

    Args:
        df (pd.DataFrame): The cleaned DataFrame with 'label' column.
        test_size (float): Proportion of data for validation.
        random_state (int): Seed for reproducibility.

    Returns:
        (pd.DataFrame, pd.DataFrame): train and validation DataFrames
    """
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['label']
    )
    return train_df, val_df


# Save the splits
def save_splits(train_df, val_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    print(f"Saved train.csv and val.csv to {output_dir}")



if __name__ == "__main__":
    INPUT_PATH = 'data/raw/tripadvisor_hotel_reviews.csv'
    OUTPUT_DIR = 'data/processed/'

    print("[INFO] Loading data...")
    df = load_data(INPUT_PATH)

    print("[INFO] Removing neutral reviews...")
    df = remove_neutral_reviews(df)

    print("[INFO] Mapping ratings to binary labels...")
    df = map_ratings_to_binary_labels(df)

    print("[INFO] Cleaning text...")
    df = clean_text(df)

    print("[INFO] Dropping empty reviews...")
    df = drop_empty_reviews(df)

    print("[INFO] Splitting into train/val...")
    train_df, val_df = split_train_val(df)

    print("[INFO] Saving splits...")
    save_splits(train_df, val_df, OUTPUT_DIR)

    print("[SUCCESS] Data prep complete!")
