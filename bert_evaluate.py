import argparse
from bert_score import score as bertscore

def main(predictions_file, references_file, output_csv=None):
    # Load summaries and references, one per line
    with open(predictions_file, 'r', encoding='utf-8') as f:
        summaries = [line.strip() for line in f if line.strip()]
    with open(references_file, 'r', encoding='utf-8') as f:
        references = [line.strip() for line in f if line.strip()]

    if len(summaries) != len(references):
        raise ValueError('Number of summaries and references must match')

    # Compute BERT scores
    P, R, F1 = bertscore(
        summaries,
        references,
        lang='en',
        rescale_with_baseline=True
    )

    # Convert to lists for saving
    p_list = [float(x) for x in P.tolist()]
    r_list = [float(x) for x in R.tolist()]
    f1_list = [float(x) for x in F1.tolist()]

    # Print overall averages
    print(f'Average BERT Precision: {sum(p_list)/len(p_list):.4f}')
    print(f'Average BERT Recall:    {sum(r_list)/len(r_list):.4f}')
    print(f'Average BERT F1:        {sum(f1_list)/len(f1_list):.4f}')

    # Optionally write detailed scores
    if output_csv:
        import csv
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['precision', 'recall', 'f1'])
            for p, r, f1 in zip(p_list, r_list, f1_list):
                writer.writerow([p, r, f1])
        print(f'Wrote detailed scores to {output_csv}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute BERTScore for summaries')
    parser.add_argument('--predictions', required=True, help='File with one summary per line')
    parser.add_argument('--references', required=True, help='File with one reference per line')
    parser.add_argument('--output_csv', help='Optional CSV file to write per-example scores')
    args = parser.parse_args()
    main(args.predictions, args.references, args.output_csv)