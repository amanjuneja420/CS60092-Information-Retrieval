import math
import sys
import numpy as np  

def read_relevance_file(file_path):
    relevance_info = {}
    with open(file_path, 'r') as file:
        for line in file:
            query_id, _, doc_id, score = line.strip().split()
            if query_id not in relevance_info:
                relevance_info[query_id] = {}
            relevance_info[query_id][doc_id] = int(score)
    return relevance_info

def compute_precision(retrieved_docs, relevant_docs, top_k):
    top_retrieved = retrieved_docs[:top_k]
    matching_docs = [doc for doc in top_retrieved if doc in relevant_docs]
    return len(matching_docs) / top_k


def avg_precision_at_k(retrieved_docs, relevant_docs, top_k):
    total_relevant = 0
    precision_sum = 0
    for idx in range(min(top_k, len(retrieved_docs))):
        if retrieved_docs[idx] in relevant_docs:
            total_relevant += 1
            precision_sum += total_relevant / (idx + 1)
    return precision_sum / top_k if top_k > 0 else 0

def ndcg_score(relevance, doc_list, top_k):
    def discounted_cumulative_gain(scores):
        return sum((2 ** score - 1) / np.log2(pos + 2) for pos, score in enumerate(scores))
    
    retrieved_scores = [relevance.get(doc_id, 0) for doc_id in doc_list[:top_k]]
    ideal_scores = sorted(relevance.values(), reverse=True)[:top_k]
    
    actual_dcg = discounted_cumulative_gain(retrieved_scores)
    ideal_dcg = discounted_cumulative_gain(ideal_scores)
    
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0

def assess_queries(results, relevance_file):
    relevance_info = read_relevance_file(relevance_file)
    
    metrics_summary = {}
    total_ap10, total_ap20, total_ndcg10, total_ndcg20 = 0, 0, 0, 0
    total_queries = len(results)
    
    for query_id, docs in results.items():
        relevant_docs = relevance_info.get(query_id, {})
        
        ap_at_10 = avg_precision_at_k(docs, relevant_docs, 10)  
        ap_at_20 = avg_precision_at_k(docs, relevant_docs, 20) 
        ndcg_at_10 = ndcg_score(relevant_docs, docs, 10)
        ndcg_at_20 = ndcg_score(relevant_docs, docs, 20)
        
        metrics_summary[query_id] = (ap_at_10, ap_at_20, ndcg_at_10, ndcg_at_20)
        
        total_ap10 += ap_at_10
        total_ap20 += ap_at_20
        total_ndcg10 += ndcg_at_10
        total_ndcg20 += ndcg_at_20

    mean_ap10 = total_ap10 / total_queries if total_queries > 0 else 0
    mean_ap20 = total_ap20 / total_queries if total_queries > 0 else 0
    avg_ndcg10 = total_ndcg10 / total_queries if total_queries > 0 else 0
    avg_ndcg20 = total_ndcg20 / total_queries if total_queries > 0 else 0
    
    return metrics_summary, mean_ap10, mean_ap20, avg_ndcg10, avg_ndcg20

def store_metrics(summary, results_file, mean_ap10, mean_ap20, avg_ndcg10, avg_ndcg20):
    output_file = results_file.replace('ranked_list', 'metrics')
    with open(output_file, 'w') as file:
        for query, (ap10, ap20, ndcg10, ndcg20) in summary.items():
            file.write(f"{query}: AP@10={ap10}, AP@20={ap20}, NDCG@10={ndcg10}, NDCG@20={ndcg20}\n")
        
        file.write("\nOverall Metrics:\n")
        file.write(f"Mean AP@10: {mean_ap10}\n")
        file.write(f"Mean AP@20: {mean_ap20}\n")
        file.write(f"Avg NDCG@10: {avg_ndcg10}\n")
        file.write(f"Avg NDCG@20: {avg_ndcg20}\n")

def main():
    relevance_file_path = sys.argv[1]
    results_file_path = sys.argv[2]
    
    query_results = {}
    with open(results_file_path, 'r') as file:
        for line in file:
            query_id, doc_str = line.strip().split(":")
            query_results[query_id] = doc_str.split()
    
    summary, mean_ap10, mean_ap20, avg_ndcg10, avg_ndcg20 = assess_queries(query_results, relevance_file_path)
    store_metrics(summary, results_file_path, mean_ap10, mean_ap20, avg_ndcg10, avg_ndcg20)

if __name__ == "__main__":
    main()
