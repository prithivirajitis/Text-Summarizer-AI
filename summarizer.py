import pandas as pd
import nltk
import sys
from evaluate import load
from datasets import load_dataset
from transformers import pipeline
import os

# --- 1. NLTK Resource Check ---
try:
    nltk.download('punkt', quiet=True) 
    nltk.download('punkt_tab', quiet=True) 
    print("NLTK 'punkt' resource ready.")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

# --- 2. Evaluation Logic ---
def evaluate_summaries(generated_summaries, reference_summaries):
    rouge = load("rouge")
    bleu = load("bleu")
    rouge_results = rouge.compute(predictions=generated_summaries, references=reference_summaries)
    bleu_results = bleu.compute(predictions=generated_summaries, references=reference_summaries)
    return {**rouge_results, "bleu": bleu_results['bleu']}

# --- 3. Data Loading (Using your Local CSV) ---
try:
    # --- OPTION A: (Commented Out) Loading from Hugging Face ---
    # dataset = load_dataset("billsum", split="ca_test") 
    
    # --- OPTION B: (ACTIVE) Loading from your local CSV ---
    print("Reading data from billsum_test_data.csv...")
    df = pd.read_csv("billsum_test_data.csv")
    
    # Define how many rows to process
    num_to_test = 5 
    
    # We extract the 'text' and 'summary' columns directly from the CSV
    sample_texts = df['text'].iloc[:num_to_test].tolist()
    reference_list = df['summary'].iloc[:num_to_test].tolist()

    print(f"CSV loaded. Processing {num_to_test} summaries...")

    # --- 4. Model Inference ---
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

    print("Generating summaries via AI model...")
    model_outputs = summarizer(sample_texts, max_length=100, min_length=30, do_sample=False)
    generated_list = [out['summary_text'] for out in model_outputs]

    # --- 5. Scoring ---
    scores = evaluate_summaries(generated_list, reference_list)

    # --- 6. Results Display ---
    print("\n" + "="*30)
    print("FINAL SCORES")
    print("="*30)
    for metric, value in scores.items():
        print(f"{metric}: {value:.4f}")

    print("\n" + "="*30)
    print("TEXT COMPARISON")
    print("="*30)
    for i in range(len(generated_list)):
        print(f"\n--- Example {i+1} ---")
        print(f"REFERENCE: {reference_list[i][:150]}...")
        print(f"AI SUMMARY: {generated_list[i]}")

except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)