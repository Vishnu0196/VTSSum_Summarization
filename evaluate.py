import os
import json
import argparse
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
from segeval import pk, window_diff
from bert_score import score as bert_score

if __name__ == '__main__':
    # Parse command-line arguments for model directory and test data folder
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True, help='Path to the SentenceTransformer model directory')
    parser.add_argument('--test_dir', required=True, help='Directory containing test JSON files')
    args = parser.parse_args()

    # Load the pre-trained SentenceTransformer model
    model = SentenceTransformer(args.model_dir)

    # Initialize a list to collect results for each file
    results = []

    # Loop over all JSON files in the test directory
    for fn in os.listdir(args.test_dir):
        if not fn.endswith('.json'):
            continue  # Skip non-JSON files

        path = os.path.join(args.test_dir, fn)
        data = json.load(open(path))  # Load file content

        # Flatten segmentation: list of lists -> single list of sentences
        sents = [s for seg in data['segmentation'] for s in seg]

        # Extract reference summary sentences labeled with 1
        refs = [item['sent']
                for clip in data['summarization'].values()
                for item in clip['summarization_data']
                if item['label'] == 1]

        # Compute segmentation evaluation metrics
        p = pk([data['segmentation']], [refs])            # Pk metric
        wd = window_diff([data['segmentation']], [refs])  # WindowDiff metric

        # Compute ROUGE-1 F1 score between reference and predicted segments
        rouge = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        r = rouge.score(' '.join(refs), ' '.join(sents))['rouge1'].fmeasure

        # Compute BERTScore precision, recall, and F1
        P, R, F = bert_score(refs, sents, model_type=args.model_dir)

        # Store results for this file
        results.append({
            'file': fn,
            'pk': p,
            'window_diff': wd,
            'rouge1': r,
            'bert_f1': float(F.mean())
        })

    # Aggregate results across all files and print the mean metrics
    df = pd.DataFrame(results)
    print(df.mean().to_dict())
