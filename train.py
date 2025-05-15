import argparse
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs_csv', required=True)
    parser.add_argument('--model_out', required=True)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    df = pd.read_csv(args.pairs_csv)
    examples = [InputExample(texts=[r['sentence1'], r['sentence2']], label=float(r['label'])) for _, r in df.iterrows()]
    loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    model = SentenceTransformer('all-MiniLM-L12-v2')
    loss = losses.CosineSimilarityLoss(model)
    model.fit([(loader, loss)], epochs=args.epochs, warmup_steps=10)
    model.save(args.model_out)