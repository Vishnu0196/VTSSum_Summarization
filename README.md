
# Video Summarization Project

This repository contains code for preprocessing lecture transcripts, generating sentence embeddings, creating contrastive pairs, fine-tuning a Sentence-BERT model, and evaluating summarization quality.

## Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/my_summarization_project.git
   cd my_summarization_project

## Dataset

This project uses the VT-SSum benchmark dataset, which provides spoken-language video transcripts and human-annotated segmentations and summaries. Clone it into your data/VT-SSum :

      git clone https://github.com/Dod-o/VT-SSum.git 

<small>

## Usage

1. Preprocess transcripts  
   `python src/preprocessing.py --input_dir data/VT-SSum/train --output_file data/cleaned/VTSSum_cleaned.csv`

2. Generate SBERT embeddings  
   `python src/embed.py --input_csv data/cleaned/VTSSum_cleaned.csv --output_embeddings data/embed/sample_embeddings.npy --sample_n 10000`

3. Create contrastive pairs  
   `python src/pair_generation.py --input_csv data/cleaned/VTSSum_cleaned_sampled.csv --embeddings data/embed/sample_embeddings.npy --output_csv data/pairs/contrastive_pairs.csv`

4. Train Sentence-BERT  
   `python src/train.py --pairs_csv data/pairs/contrastive_pairs.csv --model_out models/fine_tuned_sbert`

5. Evaluate summarization  
   `python src/evaluate.py --model_dir models/fine_tuned_sbert --test_dir data/VT-SSum/test`

6. Batch GPT evaluation  
   `python src/gpt_batch_evaluate.py --test_folder data/VT-SSum/test --output_csv results/gpt_eval.csv --batch_size 50 --start_index 0 --openai_key YOUR_OPENAI_API_KEY`

</small>
   
