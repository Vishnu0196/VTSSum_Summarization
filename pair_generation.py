import os
import argparse
import pandas as pd
import numpy as np
import random
import faiss
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

# Generate positive pairs from each segment up to a maximum
# Returns a list of (sentence1, sentence2, label)
def generate_positive_pairs(df, max_pairs_per_segment=50):
    positive_pairs = []
    for _, group in df.groupby("segment"):
        if len(group) < 2:
            continue
        sentences = group["sentence"].tolist()
        # All unique sentence pairs
        pairs = [
            (sentences[i], sentences[j])
            for i in range(len(sentences))
            for j in range(i + 1, len(sentences))
        ]
        # Sample if too many
        if len(pairs) > max_pairs_per_segment:
            pairs = random.sample(pairs, max_pairs_per_segment)
        # Label positives
        for s1, s2 in pairs:
            positive_pairs.append((s1, s2, 1))
    return positive_pairs

# Filter out sentences that are too short or trivial
def is_meaningful(sentence):
    return len(sentence.split()) > 3 and len(sentence) > 15

# Require at least `min_overlap` shared tokens
def token_overlap(s1, s2, min_overlap=2):
    tokens1 = set(s1.lower().split())
    tokens2 = set(s2.lower().split())
    return len(tokens1 & tokens2) >= min_overlap

# Generate hard-negative pairs via FAISS-based similarity search
# Returns a list of (sentence1, sentence2, label=0)
def generate_hard_negatives_faiss(
    df,
    embeddings,
    k=30,
    min_similarity=0.5,
    n_neighbors=50
):
    # 1) Filter sentences
    df_filtered = df[df["sentence"].apply(is_meaningful)].reset_index()
    # 2) Build normalized embedding matrix
    emb_filtered = normalize(
        np.array([embeddings[i] for i in df_filtered["index"]]),
        axis=1
    ).astype("float32")
    # 3) Build and query FAISS index
    index = faiss.IndexFlatIP(emb_filtered.shape[1])
    index.add(emb_filtered)
    _, neighbors = index.search(emb_filtered, n_neighbors + 1)
    # 4) Collect candidates that cross segments, pass token-overlap, and exceed similarity
    hard_negatives = []
    for i in range(len(df_filtered)):
        for j in neighbors[i][1:]:  # skip self
            if j < 0:
                continue
            if df_filtered.loc[i, "segment"] != df_filtered.loc[j, "segment"]:
                s1 = df_filtered.loc[i, "sentence"]
                s2 = df_filtered.loc[j, "sentence"]
                if not token_overlap(s1, s2):
                    continue
                sim = cosine_similarity(
                    [emb_filtered[i]], [emb_filtered[j]]
                )[0][0]
                if sim >= min_similarity:
                    hard_negatives.append((s1, s2, 0))
    # Return top-k by the order they were found
    return hard_negatives[:k]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate contrastive sentence pairs")
    parser.add_argument("--input_csv",     required=True, help="Cleaned & sampled CSV")
    parser.add_argument("--embeddings_npy",required=True, help="Embeddings .npy file")
    parser.add_argument("--output_csv",    required=True, help="Output CSV for pairs")
    parser.add_argument("--max_pos",       type=int, default=50,
                        help="Max positives per segment")
    parser.add_argument("--neg_ratio",     type=float, default=0.5,
                        help="Negative/positive ratio if --hard_k not set")
    parser.add_argument("--hard_k",        type=int,
                        help="Exact number of hard negatives (overrides neg_ratio)")
    parser.add_argument("--min_sim",       type=float, default=0.5,
                        help="Min cosine similarity for hard negatives")
    args = parser.parse_args()

    # Load inputs
    df = pd.read_csv(args.input_csv)
    embeddings = np.load(args.embeddings_npy)

    # Generate positives
    pos_pairs = generate_positive_pairs(df, args.max_pos)
    # Determine k for hard negatives
    k = args.hard_k or int(len(pos_pairs) * args.neg_ratio)
    # Generate hard negatives
    hard_pairs = generate_hard_negatives_faiss(
        df, embeddings, k=k, min_similarity=args.min_sim
    )

    # Combine and save
    all_pairs = pos_pairs + hard_pairs
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    pd.DataFrame(all_pairs, columns=["sentence1", "sentence2", "label"]) \
      .to_csv(args.output_csv, index=False)
