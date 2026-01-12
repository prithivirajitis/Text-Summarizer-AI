import pandas as pd
import nltk
import sys
from evaluate import load
from datasets import load_dataset
from transformers import pipeline

# --- 1. NLTK Resource Check ---
try:
    nltk.download('punkt', quiet=True) 
    nltk.download('punkt_tab', quiet=True) 
    print("NLTK 'punkt' resource ready.")
except Exception as e:
    print(f"Error downloading NLTK 'punkt': {e}")
    sys.exit(1)

# --- 2. Evaluation Logic (ROUGE + BLEU) ---
def evaluate_summaries(generated_summaries, reference_summaries):
    rouge = load("rouge")
    bleu = load("bleu")
    
    rouge_results = rouge.compute(predictions=generated_summaries, references=reference_summaries)
    # BLEU expects references as a list of lists for some versions, 
    # but 'evaluate' handles the standard list mapping here.
    bleu_results = bleu.compute(predictions=generated_summaries, references=reference_summaries)
    
    return {**rouge_results, "bleu": bleu_results['bleu']}

# --- 3. Data Loading & Model Inference ---
print("Downloading and loading dataset from Hugging Face Hub...")
try:
    # Load the full test split
    dataset = load_dataset("billsum", split="ca_test") 
    
    # To keep your logic but make it run efficiently for testing:
    # Let's take the first 5 examples. You can change this to len(dataset) later.
    num_to_test = 5 
    sample_texts = dataset['text'][:num_to_test]
    reference_list = dataset['summary'][:num_to_test] 

    print(f"Dataset loaded. Processing {num_to_test} summaries...")

    # Load the pre-trained BART model
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

    # Generate real summaries using the model
    print("Generating summaries via AI model...")
    model_outputs = summarizer(sample_texts, max_length=100, min_length=30, do_sample=False)
    
    # Extract text from the output dictionary
    generated_list = [out['summary_text'] for out in model_outputs]

    # --- 4. Scoring ---
    scores = evaluate_summaries(generated_list, reference_list)

    print("\n--- Evaluation Scores ---")
    for metric, value in scores.items():
        print(f"{metric}: {value:.4f}")

except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)