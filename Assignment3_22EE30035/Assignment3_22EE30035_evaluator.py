from rouge_score import rouge_scorer
import sys
import pandas as pd

def evaluator(data_file, summary_file):
    df = pd.read_csv(data_file)
    with open(summary_file, 'r', encoding='utf-8') as f:
        summaries = f.read().strip().split('*' * 50 + '\n') 
    
    rouge1_scores = []
    rouge2_scores = []

    assert len(summaries) == len(df), "Mismatch between number of documents and summaries."

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
    
    for idx, row in df.iterrows():
        original_highlight = row['highlights']  
        generated_summary = summaries[idx].split('\n', 1)[1].strip()

        scores = scorer.score(original_highlight, generated_summary)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)

        print(f"Document ID: {idx+1}")
        print(f"ROUGE-1: {scores['rouge1'].fmeasure:.4f}, ROUGE-2: {scores['rouge2'].fmeasure:.4f}")
        print("-" * 30)

    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
    
    print("\nAverage ROUGE Scores:")
    print(f"ROUGE-1: {avg_rouge1:.4f}")
    print(f"ROUGE-2: {avg_rouge2:.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Command not given correctly\n")
    else:
        data_file = sys.argv[1]
        summary_file = sys.argv[2]
        evaluator(data_file, summary_file)
