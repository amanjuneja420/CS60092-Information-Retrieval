import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize
from pulp import LpMaximize, LpProblem, LpVariable, lpSum,PULP_CBC_CMD
import nltk

nltk.download('punkt')

def calculate_relevance_scores(sentences, vectorizer):
    tfidf_matrix = vectorizer.fit_transform(sentences)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    relevance_scores = similarity_matrix.sum(axis=1)
    return relevance_scores, similarity_matrix

def calculate_redundancy_scores(similarity_matrix):
    num_sentences = similarity_matrix.shape[0]
    redundancy_scores = {}
    for i in range(num_sentences):
        for j in range(i + 1, num_sentences):
            redundancy_scores[(i, j)] = similarity_matrix[i, j]
    return redundancy_scores

def solve_ilp(relevance_scores, redundancy_scores, sentences, max_word_count, redundancy_penalty):
    num_sentences = len(sentences)
    prob = LpProblem("Sentence_Selection", LpMaximize)

    sentence_vars = [LpVariable(f"s_{i}", cat="Binary") for i in range(num_sentences)]

    pairwise_vars = {}
    for (i, j), score in redundancy_scores.items():
        pairwise_vars[(i, j)] = LpVariable(f"p_{i}_{j}", cat="Binary")

    prob += lpSum(
        sentence_vars[i] * relevance_scores[i] for i in range(num_sentences)
    ) - redundancy_penalty * lpSum(
        pairwise_vars[(i, j)] * redundancy_scores[(i, j)]
        for (i, j) in redundancy_scores
    )
    prob += lpSum(len(word_tokenize(sentences[i])) * sentence_vars[i] for i in range(num_sentences)) <= max_word_count

    for (i, j) in redundancy_scores:
        prob += pairwise_vars[(i, j)] <= sentence_vars[i]
        prob += pairwise_vars[(i, j)] <= sentence_vars[j]
        prob += pairwise_vars[(i, j)] >= sentence_vars[i] + sentence_vars[j] - 1

    prob.solve(PULP_CBC_CMD(msg=0))

    selected_sentences = [sentences[i] for i in range(num_sentences) if sentence_vars[i].value() == 1]
    return selected_sentences

def summarize_text(article, max_word_count=200, redundancy_penalty=1):
    sentences = sent_tokenize(article[0])

    vectorizer = TfidfVectorizer(stop_words='english')

    relevance_scores, similarity_matrix = calculate_relevance_scores(sentences, vectorizer)

    redundancy_scores = calculate_redundancy_scores(similarity_matrix)

    selected_sentences = solve_ilp(relevance_scores, redundancy_scores, sentences, max_word_count, redundancy_penalty)

    summary = ' '.join(selected_sentences)
    return summary

def main():
    if len(sys.argv) != 2:
        print("Command not given correctly\n")
        sys.exit(1)
    
    data_path = sys.argv[1]
    df = pd.read_csv(data_path)
    article_column = df.columns[1]

    summaries = []
    for _, row in df.iterrows():
        article = [row[article_column]]
        summary = summarize_text(article)
        summaries.append(summary)
    
    with open("Assignment3_22EE30035_summary.txt", "w", encoding="utf-8") as file:
        for idx, summary in enumerate(summaries, start=1):
            file.write(f"Document ID: {idx}\n")
            file.write("Summary:\n")
            file.write(f"{summary}\n")
            file.write("*" * 50 + "\n")

if __name__ == "__main__":
    main()
