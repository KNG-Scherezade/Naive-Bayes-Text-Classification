from __future__ import division
from codecs import open
from collections import Counter
import re
import numpy as np

multinomial_factor = 1
smoothing_factor_conditional = smoothing_factor_total = 1.0
word_character_min = 2
print_seperation_min = 0
print_seperation_max = 1000
filter = []
synonoyms = {}

global_docs_proccessed = 0
global_line_labeled = {}
global_combined_total_unqiue = 0
global_word_label_counter_dic = {}

def unique_set(labels):
    redux = []
    for label in labels:
        dupe = False
        for item in redux:
            if item == label:
                dupe = True
        if dupe:
            continue
        redux.append(label)
    return redux

def email_rule(email):
    email = email.split("@")
    return "*@" + email[1]

# https://www.analyticsvidhya.com/blog/2015/10/6-practices-enhance-performance-text-classification-model/
# https://stackoverflow.com/questions/3473612/ways-to-improve-the-accuracy-of-a-naive-bayes-classifier
def enhancements(word):
    ignore = False
    stem = False
    

    # enhancement 1 - remove case spoofing
    word = word.upper()
    
    #enhancment 2 - Email simplification
    if re.match("[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", word):
        word = email_rule(word)
    
    # enhancment 3 - word length minimum
    if len(word) < word_character_min:
        ignore = True
        return ignore, word, stem
        
    # enhancement 4 - word ignore
    # https://www.ranks.nl/stopwords
    for item in filter:
        if item.upper() == word:
            ignore = True
            return ignore, word, stem
            
    # enhancement 5 - grammatical stemming
    # https://examples.yourdictionary.com/list-of-suffixes-and-suffix-examples.html
    rule = "^([A-Z]+)(ED|ING|SION|IBLE|AL|FUL|ED|ENED|ENING|IZE|WISE)$"
    if len(word) > 3 and re.match(rule, word):
        reg = re.compile(rule)
        grou = reg.search(word)
        stem = True
        #print(grou.group(1) + " " + word)
        return ignore, re.sub(rule, grou.group(1), word), stem
    
    # enhancement 6 - Synonyms
    # https://www.englisch-hilfen.de/en/words/synonyms4.htm
    if word in synonoyms:
        word = synonoyms[word]
        pass
    
    # enhancement 7 - Spelling/Rules
    word = word.replace("EI", "IE")
    word = word.replace("(", "")
    word = word.replace(")", "")
    word = word.replace("BEGGINING", "BEGINNING")
    word = word.replace("RYTHM", "RHYTHM")
    word = word.replace("WRITEN", "WRITTEN")
    word = word.replace("WRITING", "WRITTING")
    word = word.replace("VACCUUM", "VACUUM")
    word = word.replace("TOMMORROW", "TOMORROW")
    word = word.replace("MISPELL", "MISSPELL")

    
    return ignore, word, stem
          
# https://www.youtube.com/watch?v=EGKeC2S44Rs
def bayes_multinominal(documents,labels):
    global global_word_label_counter_dic, global_line_labeled, global_docs_proccessed, global_combined_total_unqiue
    unique_labels = unique_set(labels)
    label_totals_counter = Counter()
    line_totals_counter = Counter()
    word_label_counter_dic = {}
    probability_dic = {}
    vocabulary = []
    combined_total_unqiue = 0
    word_mass_total = 0

    for label in unique_labels:
        label_totals_counter[label] = 0
        line_totals_counter[label] = 0
        word_label_counter_dic[label] = Counter()
        probability_dic[label] = Counter()
        
    # docs already word set per line
    # count things + modify things
    for index, words in enumerate(documents):
        line_totals_counter[labels[index]] += 1
        for word in words:

            drop, word, stem = enhancements(word)
            if drop:
                continue
            word_mass_total += 1
            word_label_counter_dic[labels[index]][word] += 1
            label_totals_counter[labels[index]] += 1
            if word not in probability_dic:
                #print(word)
                probability_dic[word] = {}
                combined_total_unqiue += 1
                vocabulary.append(word)
    # compute probabilities
    for label in unique_labels:
        for word in vocabulary:
            if word not in probability_dic:
                probability_dic[word] = {}
            #p(word | label) =>  log(word count + multinomial_factor) - log(label instances + vocabulary) = log(m/(n+o))  FIX
            probability_dic[word][label] = np.log(word_label_counter_dic[label][word] + multinomial_factor) - np.log(label_totals_counter[label] + combined_total_unqiue) / smoothing_factor_conditional
        #p(label)        =>  log(lines of label) - log(n) = log(m/n)     FIX
        probability_dic[label]["total"] = np.log(line_totals_counter[label] + multinomial_factor) - np.log(len(documents)) / smoothing_factor_total 
        probability_dic[label]["unknown"] = np.log(multinomial_factor) - np.log(label_totals_counter[label] + combined_total_unqiue) / smoothing_factor_total
    
    global_word_label_counter_dic = word_label_counter_dic
    global_line_labeled = label_totals_counter
    global_docs_proccessed = len(documents)
    global_combined_total_unqiue = combined_total_unqiue
    return probability_dic
    
    
def train_nb(documents, labels):
    return bayes_multinominal(documents,labels)     

def classify_documents(docs, rules, unique_labels):
    prediction_arr = []
    global global_word_label_counter_dic, global_line_labeled, global_docs_proccessed, global_combined_total_unqiue
    for words in docs:
        global_docs_proccessed += 1
        best_score = float("-inf")
        next_score = float("-inf")
        best_label = ""
        best_per_word = []
        next_word = []
        for label in unique_labels:
            independent_probability = rules[label]["total"]
            per_word = []
            for word in words:        
                drop, word, stem = enhancements(word)
                if drop:
                    continue        
                if word not in rules or label not in rules[word]:
                    independent_probability += rules[label]["unknown"]
                    per_word.append(rules[label]["unknown"])
                    pass
                else:
                    independent_probability += rules[word][label]
                    per_word.append(rules[word][label])

            if best_score < independent_probability:
                best_label = label
                best_score = independent_probability
                best_per_word = per_word
                if next_score == float("-inf"):
                    next_score = best_score
            else:
                next_score = independent_probability
                next_word = per_word
        prediction_arr.append({"guess": best_label, "best_score": best_score, "next_score": next_score})
        #prediction_arr.append({"guess": best_label})
    return prediction_arr


def accuracy(true_labels, guessed_labels, docs):
    correct = 0
    total = len(true_labels)
    total_ignored = 0
    total_incorrect = 0
    error_labels = []
    av_sepparation = 0
    for index, guess in enumerate(guessed_labels):
        if guess["guess"] == true_labels[index]:
            correct += 1
            #guess["line"] = " ".join(docs[index])
            #guess["TRUE"] = "X"
            #error_labels.append(guess)
            guess["seperation"] = np.absolute(guess["best_score"] - guess["next_score"])
            av_sepparation -= guess["seperation"]
        else:
            guess["line"] = " ".join(docs[index])
            guess["FALSE"] = "X"
            guess["seperation"] = np.absolute(guess["best_score"] - guess["next_score"])
            av_sepparation += guess["seperation"]
            if guess["seperation"] > print_seperation_min and guess["seperation"] < print_seperation_max:
                error_labels.append(guess)
            total_incorrect += 1
            pass
        if guess == "":
            total_ignored += 1
            
    #print(correct)
    #print(total)
    print("--")
    print(av_sepparation / len(guessed_labels))
    print(str(total_incorrect) + " / " + str(total) + " incorrect")
    #print(correct / (total - total_ignored))
    print(str(correct / total * 100) + "% accuracy")
    return error_labels, correct / total, total_incorrect, av_sepparation/len(guessed_labels)
# ...
# (return the accuracy)
