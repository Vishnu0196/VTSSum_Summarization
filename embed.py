import os
import argparse
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def main():
    parser = argparse.ArgumentParser(
        description='Generate embeddings for all sentences in a CSV file'
    )
    parser.add_argument(
        '--input_csv', required=True,
        help='Path to input CSV with a "sentence" column'
    )
    parser.add_argument(
        '--output_embeddings', required=True,
        help='Path to save NumPy .npy embeddings file'
    )
    parser.add_argument(
        '--output_csv', required=True,
        help='Path to save augmented CSV with embeddings'
    )
    parser.add_argument(
        '--model_name', default='all-MiniLM-L6-v2',
        help='SentenceTransformer model identifier'
    )
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='Batch size for embedding generation'
    )
    args = parser.parse_args()

    # Read all sentences
    df = pd.read_csv(args.input_csv)
    if 'sentence' not in df.columns:
        raise ValueError("Input CSV must contain a 'sentence' column")
    sentences = df['sentence'].astype(str).tolist()

    # Load model and compute embeddings
    model = SentenceTransformer(args.model_name)
    embeddings = model.encode(
        sentences,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # Ensure output directories exist
    os.makedirs(os.path.dirname(args.output_embeddings), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    # Save embeddings array
    np.save(args.output_embeddings, embeddings)

    # Attach embeddings to DataFrame and save
    df['embedding'] = embeddings.tolist()
    df.to_csv(args.output_csv, index=False)

    print(
        f"Saved {len(sentences)} embeddings to {args.output_embeddings} "
        f"and augmented CSV to {args.output_csv}"
    )


if __name__ == '__main__':
    main()
