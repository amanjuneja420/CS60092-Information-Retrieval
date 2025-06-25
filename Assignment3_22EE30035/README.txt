22EE30035
------------------------------------------------------------------------------------------------------
Python 3.12.5
Libraries used are
pandas: for handling CSV dataset.
scikit-learn: for the TF-IDF vectorizer and cosine similarity.
nltk: for sentence and word tokenization.
pulp: for solving the Integer Linear Programming (ILP) problem.
rouge-score: for calculating ROUGE-1 and ROUGE-2 scores.

---------------------------------------------------------------------------------------------------------

Design Details

SUMMARIZER PART
summarizer code generates multi-document summaries using an ILP based on relevance and redundancy. 
Below is the working explained for solving the summarization problem

1. Relevance Calculation: The code uses TF-IDF vectors and cosine similarity to calculate relevance scores for each sentence.

2. Redundancy Calculation: A pairwise similarity matrix calculates redundancy between sentences.

3. ILP Formulation: The ILP maximizes relevance while minimizing redundancy, constrained by a maximum word count (200 words).

4. Output: Summaries for each document are saved in Assignment3_22EE30035_summary.txt.

EVALUATOR PART
The evaluator code calculates ROUGE-1 and ROUGE-2 for summaries generated in task A

1. Input Files to the code: The original data file and the generated summary file.

2. Evaluation Metrics: ROUGE-1 and ROUGE-2 scores are computed for each summary.

3. Output: The ROUGE scores are displayed for each document, along with average scores for all summaries.

-------------------------------------------------------------------------------------------------------

HOW TO RUN THE CODE

THE ZIP FILE CONTAINS FIVE FILES(not necessarily named like this but named according to naming convention as in assignment)
1)SUMMARIZER.PY 
2)EVALUATOR.PY
3)DATASET-1K.CSV (DATASET ON WHICH SUMMARY WAS OBTAINED AND ROUGE SCORES WERE CALCULATED)
4)README.TXT (THIS FILE)
5)SUMMARY.TXT( OUTPUT WHEN CODE WAS RUN PREVIOUSLY )

TO RUN THE CODE OPEN TERMINAL 
WRITE python <path to summarizer.py> <path to dataset>
this will do some ilp calculations and create summary.txt

WRITE python <path to evaluator.py> <path to dataset> <path to summary.txt>

-------------------------------------------------------------------------------------------------------

Additional Information

The summarizer and evaluator scripts require the dataset to follow the CNN format and include only summaries that are within 200 words.
We use ILP based on McDonald's approach to solve multi-document summarization.
Also the code for summarizer takes atleast 30 minutes to 1 hour to run on the dataset. The running time also depending on hardware of user.
