 Everything is splitted in different files that you can import as a normal library. (For example "from src.preprocessing import *" to import all the functions that are in the file "preprocessing" located in the folder "src").

I wrote a description for each function. You can access the description by looking directly at the file or simply writing the function in the notebook and pressing shift+tab (the function must have been imported for this to work).

But basicly there are only a few functions that you might need for your work :

Cosine_sim (situated in src/classifiers.py, import with : "from src.classifiers import Cosine_sim")
    is a powerful estimator (cosine similarity from the DIS course). It works as the other estimators, such as LogisticRegression, kNeighbors... Create it with cos_clf=Cosine_sim(categories) where categories are the name of the categories in a list. Then you can use fit and predict.
parameter_search(estimator, param_grid, X_train, y_train, cv=5, scoring='accuracy', verbose=2)
    is a function that finds the best parameter for your estimator and do a nice plot. param_grid is of the format {'C':[2,3,4,5,6,7,8]} where C is the name of the parameter to tune and []2,3,4,5,6,7,8] is the values to try. Write verbose=0 if you don't want any text, or verbose=2 if you want it to print things on the way.


I use both function in my notebook project_step1_v3 so feel free to see how they work !

Send me your errors if one my function isn't working :)

