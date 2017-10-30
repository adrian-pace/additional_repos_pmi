from gensim import models,matutils


def LSI_concepts(X_train, X_test, num_lsi_topics, words):
    """
    Reduce data to concepts
    :param X_train: Data to train
    :param X_test:  Test data that will be transformed according to X_train concepts
    :param num_lsi_topics: Number of concepts to reduce to
    :param words:   Words corresponding to each feature (vectorizer.get_feature_names)
    :return: X_train_concepts, X_test_concepts
    """
    dictionary = dict()
    for i in range(len(words)):
        dictionary[i] = words[i]

    corpus = matutils.Dense2Corpus(X_train, documents_columns=False)
    lsi_transformer = models.LsiModel(corpus, id2word=dictionary, num_topics=num_lsi_topics)
    corpus_lsi = lsi_transformer[corpus]
    X_train_concepts = matutils.corpus2dense(corpus_lsi, num_terms=num_lsi_topics).T
    corpus = matutils.Dense2Corpus(X_test, documents_columns=False)
    corpus_lsi = lsi_transformer[corpus]
    X_test_concepts = matutils.corpus2dense(corpus_lsi, num_terms=num_lsi_topics).T
    return X_train_concepts, X_test_concepts


# USE LSI number of concepts and then tune update every and passes

def LDA_concepts(X_train, X_test, num_lda_topics, words,update_every=0, passes=20):
    """
    Reduce data to concepts
    :param X_train: Data to train
    :param X_test:  Test data that will be transformed according to X_train concepts
    :param num_lsi_topics: Number of concepts to reduce to
    :param words:   Words corresponding to each feature (vectorizer.get_feature_names)
    :param update_every: LDA parameter
    :param passes:  LDA parameter
    :return: X_train_concepts, X_test_concepts
    """
    dictionary = dict()
    for i in range(len(words)):
        dictionary[i] = words[i]

    corpus = matutils.Dense2Corpus(X_train, documents_columns=False)
    lsi_transformer = lda_transformer = models.LdaModel(corpus, id2word=dictionary, num_topics=num_lda_topics,
                                                        update_every=update_every, passes=passes, alpha='auto')
    corpus_lda = lda_transformer[corpus]
    X_train_concepts = matutils.corpus2dense(corpus_lda, num_terms=num_lda_topics).T
    corpus = matutils.Dense2Corpus(X_test, documents_columns=False)
    corpus_lda = lda_transformer[corpus]
    X_test_concepts = matutils.corpus2dense(corpus_lda, num_terms=num_lda_topics).T
    return X_train_concepts, X_test_concepts
