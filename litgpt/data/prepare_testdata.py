import os
import pandas as pd
from transformers import AutoTokenizer

def prepare_test_data(tokenizer_path, parquet_dir):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # List all Parquet files in the directory
    parquet_files = [
        os.path.join(parquet_dir, f)
        for f in os.listdir(parquet_dir)
        if f.endswith('.parquet')
    ]

    print(f"Found {len(parquet_files)} Parquet files in {parquet_dir}.")

    for file_path in parquet_files:
        print(f"Processing {file_path}...")

        # Read the Parquet file
        df = pd.read_parquet(file_path)

        # Concatenate "problem" and "generated_solution" into "total_text"
        df['total_text'] = df['problem'].astype(str) + " " + df['generated_solution'].astype(str)

        # Tokenize "total_text" into "tokens"
        df['tokens'] = df['total_text'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))

        # Construct new file name
        base_name = os.path.basename(file_path)
        name_without_ext = os.path.splitext(base_name)[0]
        new_file_name = f"{name_without_ext}_tokenized.parquet"
        new_file_path = os.path.join(parquet_dir, new_file_name)

        # Save the modified DataFrame
        df.to_parquet(new_file_path, index=False)
        print(f"Saved tokenized data to {new_file_path}")

    print("All files processed.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare test data by tokenizing total_text.")
    parser.add_argument("--tokenizer_path", type=str, default="/raid/longhorn/huangchen/models/TinyLlama", help="Path to the tokenizer directory.")
    parser.add_argument("--parquet_dir", type=str, default="/raid/longhorn/huangchen/Datasets/Parquet_files/val", help="Directory containing Parquet files.")

    args = parser.parse_args()

    prepare_test_data(args.tokenizer_path, args.parquet_dir)
