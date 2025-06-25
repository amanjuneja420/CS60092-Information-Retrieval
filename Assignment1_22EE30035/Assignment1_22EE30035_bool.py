# code for task C for assignment1

import pickle
import sys

def queryfile(filename):
    dict = {}
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            id = line.split("\t")[0]
            content = (line.split("\t")[1]).split()
            dict[id] = content
    return dict

def merge(sets):
    merged_set = set.intersection(*sets)
    return list(merged_set)

def retrieve_docs(inverted_index, query_list):
    sets = []
    for query in query_list:
        try:
            docs_query = set(inverted_index[query])
        except:
            docs_query = set([])
        sets.append(docs_query)
    merged_set = merge(sets)
    return sorted(list(merged_set))

def main():
    model_path = sys.argv[1]
    query_file_path = sys.argv[2]

    with open(model_path, 'rb') as f:
        inverted_index = pickle.load(f)

    query_dictionary = queryfile(query_file_path)

    with open("Assignment1_22EE30035_results.txt", "w") as f:
        for query_id, query_text in query_dictionary.items():
            merged_list = retrieve_docs(inverted_index, query_text)
            final_string = f"{query_id}:"
            for element in merged_list:
                final_string+=f"{element} "
            f.write(f"{final_string}\n")

if __name__=="__main__":
    main()
            

