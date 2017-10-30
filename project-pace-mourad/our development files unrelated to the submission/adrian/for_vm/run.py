from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from src.preprocessing import *
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier

from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer
import enchant

from src.classifiers import *
from src.CV import *

import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



random_state=0

df = pd.read_json(path_or_buf='amazon_step1.json',orient='records',lines=True)


# TODO remove sampling
# Sampling to minimze computing cost
df = df.sample(5000,random_state=random_state)
#df.head()





# create a corpus class with an iterator that reads one corpus document per line without loading all into memory

eng_dic = enchant.Dict("en_US")
tester = 1
lemmatizer = WordNetLemmatizer()
documents = np.array(df['reviewText'])

# Remove special characters
documents_no_specials = remove_specials_characters(documents)
# remove stop words and tokenize
documents_no_stop = []
for document in documents_no_specials:
    new_text = []
    for word in document.lower().split():
        if word not in STOPWORDS:
            new_text.append(word)
    documents_no_stop.append(new_text)

documents_no_stop_no_numeric = remove_numerical(documents_no_stop)

# lemmattizing tokens (better than stemming by taking word context into account)
documents_no_stop_no_numeric_lemmatize = [[lemmatizer.lemmatize(token) for token in text]
                                                    for text in documents_no_stop_no_numeric]

# remove non-english words
documents_no_stop_no_numeric_lemmatize_english = [[token for token in text if (eng_dic.check(token)) ]
                                                            for text in documents_no_stop_no_numeric_lemmatize]

# create ready corpus
df['reviewCleaned'] = [" ".join(doc) for doc in documents_no_stop_no_numeric_lemmatize_english]

vectorizer = CountVectorizer(max_df=0.4,min_df=2)

# fit vectorizer, carry out vectorization and display results
vectorizer.fit(df['reviewCleaned'])
documents_vec = vectorizer.transform(df['reviewCleaned'])

print("data loading done")

X=documents_vec.toarray()
y=df['category']
categories=np.unique(y)

# Scaling the data
X_scaled=[]
for doc in X:
    min_max_scaler = preprocessing.MinMaxScaler()
    X_scaled.append(np.ravel(min_max_scaler.fit_transform(doc.reshape(-1, 1))))

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=random_state)


# Numpy arrays
X_train=np.array(X_train)
X_test=np.array(X_test)
y_train=np.array(y_train)
y_test=np.array(y_test)


print("Starting classifying")
print("Dummy")

dmy=DummyClassifier(random_state=random_state)
dmy.fit(X_train,y_train)
y_pred=dmy.predict(X_test)
print("accuracy Dummy Classifier:",accuracy_score(y_test,y_pred))

print("cosine_sim")


start= time.time()
cosine_classifier=Cosine_sim(categories)
print("Mean CV valdiation score for Cosine_sim:",np.mean(cross_val_score(
    cosine_classifier,X_train,y_train,scoring='accuracy',cv=5,verbose=2)))
end=time.time()
print("Execution time : {} seconds".format(end-start))


print("Decision_tree")
parameter_search(DecisionTreeClassifier(random_state=random_state),
                {'max_depth':range(1,200,10)},
                X_train,
                y_train, n_jobs=-1,save_file='output_figs/decision_tree.jpg')

print("KNeighborsClassifiers")


parameter_search(KNeighborsClassifier(),
                {'n_neighbors':range(1,50,3)},
                X_train,
                y_train,save_file='output_figs/knn.jpg')

print("SVC")
parameter_search(SVC(),
                {'C':np.logspace(-5,5,11)},
                X_train,
                y_train,save_file='output_figs/SVC.jpg')

print("logistic regression")
parameter_search(LogisticRegression(),
                {'C':np.logspace(-5,5,11)},
                X_train,
                y_train,save_file='output_figs/logreg.jpg')

