import os
import json
import pandas as pd
import argparse
from tqdm import tqdm


def preprocess(input_dir, output_file):
    """
    Reads JSON lecture files from the input directory, extracts sentences and summary labels,
    and writes a consolidated CSV to the output file.

    Each row in the output contains:
      - transcript_id: lecture identifier
      - segment: segment index within the lecture
      - sentence: cleaned sentence text
      - is_summary: binary label (1 if part of summary, else 0)
    """
    rows = []

    # Collect all JSON filenames in the input directory
    files = [f for f in os.listdir(input_dir) if f.endswith('.json')]

    # Iterate over each lecture file to extract sentences
    for fn in files:
        file_path = os.path.join(input_dir, fn)
        with open(file_path, encoding='utf-8') as f:
            lecture = json.load(f)

        # Get lecture ID, default to empty string if missing
        lid = lecture.get('id', '')

        # Loop over each segment and sentence
        for seg_idx, segment in enumerate(lecture.get('segmentation', [])):
            for sent in segment:
                # Clean and normalize sentence text
                txt = sent.strip().replace("\n", " ")

                # Skip unusually short sentences
                if len(txt) < 3:
                    continue

                # Add sentence record to rows
                rows.append({
                    'transcript_id': lid,
                    'segment': seg_idx,
                    'sentence': txt,
                    'is_summary': 0  # placeholder, will update later
                })

    # Create DataFrame from extracted sentences
    df = pd.DataFrame(rows)

    # Build a set of reference summary sentences
    summary = set()
    for fn in files:
        file_path = os.path.join(input_dir, fn)
        with open(file_path, encoding='utf-8') as f:
            lecture = json.load(f)

        # Each clip may contain summarization samples
        for clip in lecture.get('summarization', {}).values():
            if clip.get('is_summarization_sample'):
                for item in clip.get('summarization_data', []):
                    # Collect sentences labeled as summary
                    if item['label'] == 1:
                        summary.add(item['sent'].strip())

    # Mark sentences in DataFrame that appear in the summary set
    df['is_summary'] = df['sentence'].apply(lambda s: int(s in summary))

    # Save the final annotated DataFrame to CSV
    df.to_csv(output_file, index=False)


if __name__ == '__main__':
    # Parse command-line arguments for input directory and output CSV path
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='Directory containing lecture JSON files')
    parser.add_argument('--output_file', required=True, help='Path to save the output CSV')
    args = parser.parse_args()

    # Run preprocessing pipeline
    preprocess(args.input_dir, args.output_file)