from collections import Counter
import bayes
import json

def read_documents(doc_file):
    docs = []
    labels = []
    with open(doc_file, encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            docs.append(words[3:])
            labels.append(words[1])
    return docs, labels

def stopword_eval(stopwords):
    for i in reversed(range(0, len(stopwords))):
        removed_item = stopwords[i]
        stopwords.remove(removed_item)
        
        classification_rules = bayes.train_nb(train_docs, train_labels)
        best_guess = bayes.classify_documents(eval_docs, classification_rules, bayes.unique_set(train_labels))
        error_cases,acc, incorrects, seperation = bayes.accuracy(eval_labels, best_guess, eval_docs)
        log_file = open("stopword_eval.txt", "a+")
        log_file.write(str(acc) + " " + str(incorrects) + " " + " ".join(stopwords) + "\n" )
        log_file.close()

def synonym_eval():
    # manually set max range each time
    for i in range(0, 430):
        bayes.synonoyms = {}
        with open("synonyms.txt", 'r') as syn_file:
            raw_synonoyms = syn_file.readlines()
            for index,line in enumerate(raw_synonoyms):
                if index > i:
                    line = line.replace("\n", "").upper().split("\t")
                    lhs = line[1].split(",")
                    rhs = line[0]
                    for key in lhs:
                        bayes.synonoyms[key] = rhs   
        classification_rules = bayes.train_nb(train_docs, train_labels)
        best_guess = bayes.classify_documents(eval_docs, classification_rules, bayes.unique_set(train_labels))
        error_cases,acc, incorrects, seperation = bayes.accuracy(eval_labels, best_guess, eval_docs)
        log_file = open("synonym_eval.txt", "a+")
        log_file.write(str(acc) + " " + str(incorrects) + " " + " ".join(json.dumps(bayes.synonoyms)) + "\n" )
        log_file.close()
        
def smoothing_eval():
    #best implies least incorrect
    best_one = float("inf")
    best_two = 0
    bayes.multinomial_factor = 0.5
    smoothing_increment = 0.05
    while True:
        best_two = best_one

        bayes.multinomial_factor += smoothing_increment
        
        classification_rules = bayes.train_nb(train_docs, train_labels)
        best_guess = bayes.classify_documents(eval_docs, classification_rules, bayes.unique_set(train_labels))
        error_cases,acc, incorrects, seperation = bayes.accuracy(eval_labels, best_guess, eval_docs)
        best_one = incorrects 
        log_file = open("smoothing_eval.txt", "a+")
        log_file.write(str(acc) + " " + str(seperation) + " " + str(incorrects) + " " + str(bayes.multinomial_factor) + "\n" )
        log_file.close()
        print(bayes.multinomial_factor)
        print(str(best_one) + " " + str(best_two))
        if best_one > best_two:
            continue

if __name__ == "__main__":
    v = input()
    if v == "":
        all_docs, all_labels = read_documents('all_sentiment_shuffled.txt')
        split_point = int(0.80*len(all_docs))
        train_docs = all_docs[:split_point]
        train_labels = all_labels[:split_point]
        eval_docs = all_docs[split_point:]
        eval_labels = all_labels[split_point:]
    else:
        train_docs, train_labels = read_documents('all_sentiment_shuffled.txt')
        eval_docs, eval_labels = read_documents(v)
    with open("stopwords.txt", 'r') as stop_file:
        bayes.filter = stop_file.readlines()
        for index,line in enumerate(bayes.filter):
            bayes.filter[index] = line.replace("\n", "").upper()
    with open("synonyms.txt", 'r') as syn_file:
        raw_synonoyms = syn_file.readlines()
        for index,line in enumerate(raw_synonoyms):
            line = line.replace("\n", "").upper().split("\t")
            lhs = line[1].split(",")
            rhs = line[0]
            for key in lhs:
                bayes.synonoyms[key] = rhs
    print(bayes.synonoyms)
    #log_file = open("stopword_eval.txt", "a+")
    
    classification_rules = bayes.train_nb(train_docs, train_labels)
    best_guess = bayes.classify_documents(eval_docs, classification_rules, bayes.unique_set(train_labels))
    error_cases,acc, incorrects, seperation = bayes.accuracy(eval_labels, best_guess, eval_docs)
    
    #log_file.write(str(acc) + " " + str(incorrects) + " " + " ".join(bayes.filter)  + "\n")
    #log_file.close()
    
    for error in error_cases:
        print(error)
        pass
    
    #print(len(error_cases))
    #smoothing_eval()
        