import re

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