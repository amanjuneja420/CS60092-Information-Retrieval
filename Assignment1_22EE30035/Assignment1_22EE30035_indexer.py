# code for Task A of assignment 1

import re
import nltk
import pickle

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import WordNetLemmatizer
import sys

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens

def file_parser(file_path):
    documents = {}
    current_id = None
    capture_text = False
    current_text = []

    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('.I'):
                if current_id:
                    documents[current_id] = ' '.join(current_text)
                current_id = line.split()[1]
                current_text = []
                capture_text = False
            elif line.startswith('.W'):
                capture_text = True
            elif line.startswith('.X'):  
                capture_text = False
            elif capture_text:
                current_text.append(line.strip())

        if current_id:
            documents[current_id] = ' '.join(current_text)

    return documents

def main():
    file_path = sys.argv[1]
    
    documents = file_parser(file_path)
    
    inverted_index = {}

   
    for doc_id, text in documents.items():

        tokens = preprocess_text(text)
            
        for token in tokens:
            if token not in inverted_index:
                inverted_index[token] = set()
            inverted_index[token].add(doc_id)
    
    # snippet to calculate the vocabulary length of corpus
    # vocab = set()
    # for doc_id,text in documents.items():
    #     alpha = re.sub(r'[^\w\s]', '', text)
    #     alpha = alpha.lower()
    #     vocab.update(alpha.split())
    # print("vocabulary length of corpus = ",len(vocab))

    with open("model_queries_22EE30035.bin", 'wb') as f:
            pickle.dump(inverted_index, f)

if __name__ == "__main__":
    main()

