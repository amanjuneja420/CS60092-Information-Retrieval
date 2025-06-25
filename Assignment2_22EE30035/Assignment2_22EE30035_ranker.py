import re
import math
import pickle
import sys
import csv
import os
import xml.etree.ElementTree as ET
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text) 
    tokens = word_tokenize(text.lower()) 
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  
    return tokens

def parse_cord19_documents(file_path):
    documents = {}
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            cord_uid = row['cord_uid']
            abstract = row['abstract'].strip()
            if abstract: 
                documents[cord_uid] = abstract
    return documents

def build_inverted_index(documents):
    inverted_index = {}
    doc_max_tf = {} 
    doc_count = len(documents)

    for doc_id, text in documents.items():
        tokens = preprocess_text(text)
        token_freq = {}
        for token in tokens:
            if token not in inverted_index:
                inverted_index[token] = {}
            if doc_id not in inverted_index[token]:
                inverted_index[token][doc_id] = 0
            inverted_index[token][doc_id] += 1

            if token not in token_freq:
                token_freq[token] = 0
            token_freq[token] += 1
        
        doc_max_tf[doc_id] = max(token_freq.values())

    return inverted_index, doc_max_tf, doc_count

def calculate_avg_tf(documents):
    total_tf = 0
    total_terms = 0

    for text in documents.values():
        tokens = preprocess_text(text)
        total_tf += sum(tokens.count(token) for token in set(tokens))
        total_terms += len(set(tokens))
    
    avg_tf = total_tf / total_terms if total_terms != 0 else 0
    return avg_tf

def save_inverted_index(inverted_index, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(inverted_index, f)

def load_inverted_index(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def calculate_tf_idf_weighting(inverted_index, doc_max_tf, avg_tf, doc_count, scheme):
    tf_idf = {}

    for term, postings in inverted_index.items():
        df = len(postings)
        idf = math.log(doc_count / df) if scheme in ['lnc.ltc', 'lnc.Ltc', 'anc.apc'] else 1  

        for doc_id, tf in postings.items():
            max_tf_doc = doc_max_tf[doc_id] 

            if scheme.startswith('lnc'):
                tf_weight_doc = 1 + math.log(tf) if tf > 0 else 0  
            elif scheme.startswith('anc'):
                tf_weight_doc = 0.5 + 0.5 * (tf / max_tf_doc) 

            if scheme.endswith('.ltc'):
                tf_weight_query = 1 + math.log(tf) 
            elif scheme.endswith('.Ltc'):
                tf_weight_query = (1 + math.log10(tf)) / (1 + math.log10(avg_tf)) if avg_tf > 0 else 0  
            elif scheme.endswith('.apc'):
                tf_weight_query = tf  

            tf_idf_value = tf_weight_doc * tf_weight_query * idf

            if doc_id not in tf_idf:
                tf_idf[doc_id] = {}
            tf_idf[doc_id][term] = tf_idf_value

    return tf_idf

def cosine_similarity(query_vector, document_vector):
    dot_product = sum(query_vector[t] * document_vector.get(t, 0) for t in query_vector)
    query_magnitude = math.sqrt(sum(v**2 for v in query_vector.values()))
    doc_magnitude = math.sqrt(sum(v**2 for v in document_vector.values()))
    return dot_product / (query_magnitude * doc_magnitude) if query_magnitude * doc_magnitude != 0 else 0

def rank_documents(tf_idf_matrix, query_vector):
    scores = {}
    for doc_id, document_vector in tf_idf_matrix.items():
        score = cosine_similarity(query_vector, document_vector)
        scores[doc_id] = score
    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in ranked_docs[:50]]

def parse_queries(file_path, inverted_index, doc_count):
    queries = {}
    tree = ET.parse(file_path)
    root = tree.getroot()

    idfs = {}
    for term, postings in inverted_index.items():
        df = len(postings)
        idfs[term] = math.log(doc_count / df) if df > 0 else 0

    for topic in root.findall('topic'):
        query_id = topic.get('number') 
        query_element = topic.find('query')

        if query_id is not None and query_element is not None:
            query_text = query_element.text.strip()
            tokens = preprocess_text(query_text)
            query_vector = {}
            for token in tokens:
                tf = tokens.count(token)
                tf_weight = 1 + math.log(tf) if tf > 0 else 0 

                idf_weight = idfs.get(token, 0)
                query_vector[token] = tf_weight * idf_weight

            queries[query_id] = query_vector
    
    return queries

def save_ranked_results(ranked_results):
    roll_no = "22EE30035"
    for scheme, ranked_docs in ranked_results.items():
        if scheme == "lnc.ltc":
            file_name = f"Assignment2_{roll_no}_ranked_list_A.txt"
        elif scheme == "lnc.Ltc":
            file_name = f"Assignment2_{roll_no}_ranked_list_B.txt"
        elif scheme == "anc.apc":
            file_name = f"Assignment2_{roll_no}_ranked_list_C.txt"

        with open(file_name, 'w') as f:
            for query_id, docs in ranked_docs.items():
                f.write(f"{query_id}: {' '.join(docs)}\n")


def main():
    data_folder = sys.argv[1]  
    model_path = sys.argv[2]    

    cord19_file_path = os.path.join(data_folder, "cord19_dataset.csv") 
    query_file_path = os.path.join(data_folder, "topics-rnd5.xml")      

    if not os.path.isfile(query_file_path):
        sys.exit(1)

    documents = parse_cord19_documents(cord19_file_path)
    inverted_index, doc_max_tf, doc_count = build_inverted_index(documents)
    save_inverted_index(inverted_index, model_path)
    avg_tf = calculate_avg_tf(documents)

    try:
        inverted_index = load_inverted_index(model_path)
        doc_count = len(inverted_index)  
    except FileNotFoundError:
        pass

    queries = parse_queries(query_file_path, inverted_index, doc_count)
    schemes = ["lnc.ltc", "lnc.Ltc", "anc.apc"] 

    ranked_results = {}
    for scheme in schemes:
        tf_idf_matrix = calculate_tf_idf_weighting(inverted_index, doc_max_tf, avg_tf, doc_count, scheme)
        ranked_results[scheme] = {query_id: rank_documents(tf_idf_matrix, query_vector) for query_id, query_vector in queries.items()}
    save_ranked_results(ranked_results)


if __name__ == "__main__":
    main()
