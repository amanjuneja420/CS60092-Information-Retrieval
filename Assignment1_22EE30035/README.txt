22EE30035

Python version - Python 3.12.5
Libraries/Dependencies used - pickle , sys , re , nltk

Details for Design of pipeline - 
1)  Preprocessing of text: 
	-getting list of stopwords and remove stopwords from corpus.
	-tokenisation of words using word_tokenize and lemmatization using WordNetLemmatizer from nltk.corpus

2) creation of inverted index
	- built using python's dictionary data structure where keys are the token and values are associated documents with the token

3) the output is presented in sorted manner in order of document id's and the query id's are also sorted for a good presentation of results 

4) for query handling we used a custom merge function to find documents intersection of lists is computed

5) for handling of errorneous cases i.e. where there is no doc id for a given word we assign an empty set such that program doesnot fail for such cases

6) vocabulary of whole corpus = 10752 unique words/terms.