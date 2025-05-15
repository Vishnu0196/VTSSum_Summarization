#!/usr/bin/env python3
"""
evaluate_batch.py

Batch-process JSON lecture files: generate segments & centroid summaries,
then have GPT rate them. Outputs results to a CSV.
"""

import os
import json
import csv
import argparse

import numpy as np
import openai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load your fine-tuned SBERT model
model = SentenceTransformer('models/fine_tuned_sbert')

def evaluate_with_gpt(transcript, segments, summaries, model_name='gpt-3.5-turbo'):
    """
    Send transcript + segments + summaries to OpenAI and get back ratings 1–5
    on coherence, coverage, and fluency with brief justifications.
    """
    prompt = (
        'You are a helpful assistant evaluating segmentation and summaries.\n\n'
        'Transcript excerpt:\n' + transcript[:1000] + '\n\n'
        'Segments with summaries:\n'
    )
    for i, (seg, summ) in enumerate(zip(segments, summaries), start=1):
        prompt += f"-- Segment {i}: {seg[:300]}\n   Summary: {summ[:150]}\n"

    prompt += (
        '\nFor each of the following, score 1–5 and justify succinctly:\n'
        '1. Segment coherence\n'
        '2. Summary coverage\n'
        '3. Summary fluency\n'
    )

    resp = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {'role': 'system', 'content': 'You evaluate summaries.'},
            {'role': 'user',  'content': prompt}
        ],
        temperature=0.3,
        max_tokens=512
    )
    return resp.choices[0].message.content.strip()

def run_in_batches(test_folder, output_csv, batch_size=50, start_index=0, model_name='gpt-3.5-turbo'):
    files = sorted(f for f in os.listdir(test_folder) if f.lower().endswith('.json'))
    end = min(start_index + batch_size, len(files))

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    header_needed = not os.path.exists(output_csv) or os.stat(output_csv).st_size == 0

    with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if start_index == 0 and header_needed:
            writer.writerow(['filename', 'gpt_eval'])

        for idx in range(start_index, end):
            fn = files[idx]
            path = os.path.join(test_folder, fn)
            print(f"[{idx+1}/{len(files)}] Processing {fn}...")

            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # flatten transcript into sentences
            segments = data.get('segmentation', [])
            sentences = [s for seg in segments for s in seg]
            transcript = ' '.join(sentences)

            # compute sentence embeddings
            embeddings = model.encode(sentences)

            # detect boundaries where similarity dips below 0.8
            boundaries = [
                i+1 for i in range(len(embeddings)-1)
                if cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0] < 0.8
            ]

            # build segments
            seg_texts = []
            start = 0
            for b in boundaries + [len(sentences)]:
                seg_texts.append(' '.join(sentences[start:b]))
                start = b

            # generate centroid-based top-3 summaries for first 5 segments
            summaries = []
            for seg in seg_texts[:5]:
                if not seg:
                    summaries.append('')
                    continue
                seg_sents = seg.split(' ')
                # find corresponding embeddings slice
                count = len(seg_sents)
                emb_slice = embeddings[start - count : start]
                centroid = np.mean(emb_slice, axis=0).reshape(1, -1)
                sims = cosine_similarity(emb_slice, centroid).flatten()
                top3 = sims.argsort()[-3:][::-1]
                summary = ' '.join(seg_sents[i] for i in sorted(top3))
                summaries.append(summary)

            # get GPT evaluation
            rating = evaluate_with_gpt(transcript, seg_texts[:5], summaries, model_name=model_name)
            writer.writerow([fn, rating])

def main():
    parser = argparse.ArgumentParser(description='Batch GPT-based evaluation of lecture JSONs')
    parser.add_argument('--test_folder',  required=True, help='Folder of input JSON files')
    parser.add_argument('--output_csv',   required=True, help='CSV file for output')
    parser.add_argument('--batch_size',   type=int,   default=50, help='How many files to process')
    parser.add_argument('--start_index',  type=int,   default=0,  help='Index to start from')
    parser.add_argument('--openai_key',   required=True, help='Your OpenAI API key')
    parser.add_argument('--model_name',   default='gpt-3.5-turbo', help='OpenAI model for evaluation')
    args = parser.parse_args()

    openai.api_key = args.openai_key
    run_in_batches(
        test_folder=args.test_folder,
        output_csv=args.output_csv,
        batch_size=args.batch_size,
        start_index=args.start_index,
        model_name=args.model_name
    )

if __name__ == '__main__':
    main()
