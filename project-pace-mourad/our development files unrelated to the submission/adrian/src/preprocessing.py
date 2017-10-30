import re
import numpy as np

def remove_specials_characters(documents):
    """
    Remove special characters
    :param documents: the documents from where we remove the characters
    :return: the documents without the special characters
    """
    documents_no_specials = []
    for item in documents:
        documents_no_specials.append(
            item.replace('\r', ' ').replace('/n', ' ').replace('.', ' ').replace(',', ' ').replace('(', ' ') \
            .replace(')', ' ').replace("'s", ' ').replace('"', ' ') \
            .replace('!', ' ').replace('?', ' ').replace("'", '') \
            .replace('>', ' ').replace('$', ' ') \
            .replace('-', ' ').replace(';', ' ') \
            .replace(':', ' ').replace('/', ' ').replace('#', ' '))
    return documents_no_specials


def remove_numerical(documents):
    """
    remove the words containing numbers
    :param documents: 
    :return: 
    """
    def hasNumbers(inputString):
        return bool(re.search(r'\d', inputString))

    documents_no_stop_no_numeric = [[token for token in text if not (hasNumbers(token))]
                                    for text in documents]
    return documents_no_stop_no_numeric


def keepAdjectives(documents,NEGATIONWORDS, POSITIVEWORDS, NEGATIVEWORDS,negation_distance_threshold):
    document_neg_pos = []
    document_neg_count = np.zeros(documents.shape[0])
    document_pos_count = np.zeros(documents.shape[0])
    for idx,row in enumerate(documents):
        review = []
        new_word = ""
        negation_distance=0
        for word in row.split(" "):
            if len(new_word) != 1:
                negation_distance+=1
                if negation_distance>negation_distance_threshold:
                    new_word=""
            if word in NEGATIONWORDS:
                new_word = word
                negation_distance=0
            elif word in POSITIVEWORDS:
                if len(new_word)==0:
                    review.append(word)
                    document_pos_count[idx]+=1
                else:
                    #print(idx)
                    document_neg_count[idx]+=1
                    new_word = new_word + word
                    review.append(new_word)
                    new_word = ""
            elif word in NEGATIVEWORDS:
                if len(new_word)==0:
                    review.append(word)
                    document_neg_count[idx]+=1
                else:
                    #print(idx)
                    document_pos_count[idx]+=1
                    new_word = new_word + word
                    review.append(new_word)
                    new_word = ""
        document_neg_pos.append(review)
    return document_neg_pos, document_pos_count, document_neg_count