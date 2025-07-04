Roll Number: 22EE30035

Python Environment and Libraries:
---------------------------------
Python Version: 3.12.5

Libraries Used:
---------------
- nltk
- re
- math
- pickle
- sys
- csv
- os
- xml.etree.ElementTree
- numpy

Note: The `nltk` library is used for tokenization, stopword removal, and lemmatization, which are essential for processing the text. Specifically, `nltk.download('punkt')`, `nltk.download('stopwords')`, and `nltk.download('wordnet')` are included to ensure proper functionality during preprocessing.

--------------------------------------------------

Design Details:
---------------

1. Inverted Index Construction:
   - The corpus is preprocessed using tokenization, stopword removal, and lemmatization.
   - An inverted index is built where each term in the corpus is mapped to the documents containing the term, along with the frequency of the term in each document.

2. Weighting Schemes:
   - Three weighting schemes are implemented for ranking the documents:
     - lnc.ltc
     - lnc.Ltc
     - anc.apc
   - These schemes use term frequency (TF) and inverse document frequency (IDF) to compute weights for both documents and queries. Logarithmic weighting (lnc, ltc) or absolute count (anc, apc) is applied based on the scheme.

3. Ranking Process:
   - Cosine similarity is used to rank documents based on the computed TF-IDF values.
   - For each query, the documents with the highest similarity scores are ranked, and the top 50 documents are returned.

4. Metrics Computation:
   - Evaluation metrics such as AP@10, AP@20, NDCG@10, and NDCG@20 are calculated using the relevance judgments.
   - Precision, Average Precision (AP), Discounted Cumulative Gain (DCG), and Normalized DCG (NDCG) are used to measure ranking effectiveness.

--------------------------------------------------

Dataset Version:
----------------
- Dataset: A smaller version of the CORD-19 dataset.
  - Format: cord_uid abstracts (documents).
  - Download link: https://drive.google.com/file/d/1yE_eyCWI336ELjDO9Ysylgkxy0ip3pN8/view?usp=sharing
  
- Relevance file: https://ir.nist.gov/trec-covid/data/qrels-covid_d5_j0.5-5.txt
- Query file: https://ir.nist.gov/trec-covid/data/topics-rnd5.xml

--------------------------------------------------

File Descriptions:
------------------

1. `Assignment2_22EE30035_ranker.py`:
   - This script implements the ranker. It processes the corpus, builds the inverted index, parses queries, and ranks documents based on specified weighting schemes.

2. `Assignment2_22EE30035_evaluator.py`:
   - This script implements the evaluator. It computes evaluation metrics (AP@10, AP@20, NDCG@10, NDCG@20) using the relevance judgments and ranked lists.

3. `Assignment2_22EE30035_metrics_A.txt`, `Assignment2_22EE30035_metrics_B.txt`, `Assignment2_22EE30035_metrics_C.txt`:
   - These files contain the evaluation results (AP@10, AP@20, NDCG@10, NDCG@20) for the ranking schemes `lnc.ltc`, `lnc.Ltc`, and `anc.apc`, respectively.

4. `model_queries_22EE30035.bin`:
   - This binary file stores the inverted index generated by `ranker.py` and is used for subsequent runs.
   - This file is passed as a command-line input when running `ranker.py`.

--------------------------------------------------

Vocabulary Length and Preprocessing:
------------------------------------
- Vocabulary Length: 16488 (calculated during inverted index creation).
- The vocabulary is generated by tokenizing the corpus, removing stopwords, and applying lemmatization.
- Special characters and punctuation are removed during preprocessing to standardize the input data.

--------------------------------------------------

How to Run the Code:
--------------------

1. **Setting Up the Data:**
   - Create a folder named `data` in the same directory as `Assignment2_22EE30035`.
   - In the `data` folder, add the following files:
     - `cord19_dataset.csv` (the dataset file).
     - `topics-rnd5.xml` (the query file).
     - `qrels-covid_d5_j0.5-5.txt` (the relevance judgments file).

2. **Running Task A (Ranker):**
   - Open the terminal and navigate to the `Assignment2_22EE30035` directory.
   - Run the ranker with the following command:
     ```
     python <path_to_ranker.py> <path_to_data_folder> <path_to_model_queries_file>
     ```
   - The model queries file should be named `model_queries_<Roll_No>.bin` and is created during the ranker execution.

3. **Running Task B (Evaluator):**
   - Open the terminal and navigate to the `Assignment2_22EE30035` directory.
   - Run the evaluator with the following command:
     ```
     python <path_to_evaluator.py> <path_to_relevance_judgment_file> <path_to_ranked_list_K.txt>
     ```
   - Here, `K` refers to A, B, or C based on the weighting scheme you are evaluating.
   - For example, use `ranked_list_A.txt` for the `lnc.ltc` scheme, `ranked_list_B.txt` for the `lnc.Ltc` scheme, and `ranked_list_C.txt` for the `anc.apc` scheme.

--------------------------------------------------

Additional Information:
------------------------
- The entire process, from inverted index construction to query parsing and document ranking, is implemented without external libraries like Lucene or Elasticsearch.
- Only basic Python libraries and `nltk` are used for text processing.
- The results are saved in the specified format, and the ranking is evaluated based on the provided queries and relevance judgments.
