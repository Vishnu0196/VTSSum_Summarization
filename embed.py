import os
import argparse
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def generate_embeddings(input_csv, output_npy, sample_n):
    """
    Reads a CSV of sentences with summary labels, samples summary sentences,
    computes embeddings using a SentenceTransformer model, and saves both the
    embeddings array and an updated CSV with embeddings.

    Args:
        input_csv (str): Path to input CSV containing 'sentence' and 'is_summary' columns.
        output_npy (str): Path to save the numpy array of embeddings (.npy).
        sample_n (int): Number of summary sentences to sample for embedding.
    """
    # Load the CSV into a DataFrame
    df = pd.read_csv(input_csv)

    # Filter to sentences marked as summary and sample a subset
    df_summary = df[df['is_summary'] == 1].sample(n=sample_n, random_state=42)

    # Extract sentences for embedding
    sentences = df_summary['sentence'].tolist()

    # Load the pretrained embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Compute embeddings with a progress bar
    embeddings = model.encode(sentences, show_progress_bar=True)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_npy), exist_ok=True)

    # Save embeddings array to disk
    np.save(output_npy, embeddings)

    # Add embeddings to DataFrame and save updated CSV
    df_summary = df_summary.reset_index(drop=True)
    df_summary['embedding'] = embeddings.tolist()
    sampled_csv = input_csv.replace('.csv', '_sampled.csv')
    df_summary.to_csv(sampled_csv, index=False)


if __name__ == '__main__':
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(description='Generate sentence embeddings for summary samples')
    parser.add_argument('--input_csv', required=True, help='Path to the input CSV file')
    parser.add_argument('--output_embeddings', required=True, help='Path to output .npy embeddings file')
    parser.add_argument('--sample_n', type=int, default=10000, help='Number of summary sentences to sample')
    args = parser.parse_args()

    # Generate and save embeddings
    generate_embeddings(args.input_csv, args.output_embeddings, args.sample_n)
