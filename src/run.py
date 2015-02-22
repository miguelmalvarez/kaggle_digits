import pandas as pd
import numpy as np
import logging
import time
import datetime
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.grid_search import GridSearchCV

# TODO: Use pipelines
# TODO: Learning curves

# Formated current timestamp
def current_timestamp():
    ts = time.time()
    return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

# Log message with timestamp
def log_info(message):
    ts = time.time()
    logging.info(message + " " + current_timestamp())

# Force a symlink overwriting it if it already exists
def force_symlink(file1, file2):
    try:
        os.symlink(file1, file2)
    except e:
        if e.errno == errno.EEXIST:
            os.remove(file2)
            os.symlink(file1, file2)

# Initialise logging
def init_logging(log_file_path, log_file_name):
    file_path = log_file_path + log_file_name
    logging.basicConfig(format='%(message)s', level=logging.INFO, filename=file_path)
    force_symlink(file_path, 'last_run')

# List of candidate family classifiers with parameters for grid search
# [name, classifier object, parameters].
def candidate_families():
    candidates = []
    svm_tuned_parameters = [{'kernel': ['poly'], 'degree': [3]}]
    candidates.append(["SVM", SVC(C=1), svm_tuned_parameters])
    rf_tuned_parameters = [{"n_estimators": [1000]}]
    candidates.append(["RandomForest", RandomForestClassifier(n_jobs=-1), rf_tuned_parameters])        
    knn_tuned_parameters = [{"n_neighbors": [3, 5, 10]}]
    candidates.append(["kNN", KNeighborsClassifier(), knn_tuned_parameters])
    return candidates

# Fitting a feature selector 
def feature_selection(train_instances):
    log_info('Crossvalidation started... ') 
    selector = VarianceThreshold()
    selector.fit(train_instances)
    log_info('Number of features used... ' + str(Counter(selector.get_support())[True]))
    log_info('Number of features ignored... ' + str(Counter(selector.get_support())[False]))
    return selector

# Return models with quality estimations from a set of model families given training data using crosvalidation
def quality_models(classifier_families, train_instances, judgements):
    best_quality = 0.0
    best_classifier = None    
    classifiers = []
    for name, model, parameters in classifier_families:
        log_info('Grid search for... ' + name)
        clf = GridSearchCV(model, parameters, cv=5, scoring="accuracy", verbose=5, n_jobs=4)
        clf.fit(train_instances, judgements)
        best_estimator = clf.best_estimator_
        log_info('Best hyperparameters: ' + str(clf.best_params_))
        classifiers.append([str(clf.best_params_), clf.best_score_, best_estimator])

    return sorted(classifiers, key=lambda model: model[1], reverse=True);

# Returns the best model from a set of model families given  training data using crosvalidation
def best_model(classifier_families, train_instances, judgements):
    models = quality_models(classifier_families, train_instances, judgements)
    best = models[0]
    log_info('Best model: ' + str(best));
    return best[2]

# Run the data and over multiple classifier and output the data in a csv file using a 
# specific scaling object
def run(scaler, output_path):    
    log_info('============== \nClassification started... ')

    log_info('Reading training data... ')
    train_data = pd.read_csv('data/train-sample.csv', header=0).values
    #the first column of the training set will be the judgements
    judgements = np.array([str(int (x[0])) for x in train_data])
    train_instances = np.array([x[1:] for x in train_data])
    train_instances = [[float(x) for x in instance] for instance in train_instances]

    log_info('Reading testing data... ')
    test_data = pd.read_csv('data/test-sample.csv', header=0).values
    test_instances = np.array([x[0:] for x in test_data])
    test_instances = [[float(x) for x in instance] for instance in test_instances]
    
    #Feature selection
    logging.info("Selecting features... ")
    fs = feature_selection(train_instances)
    train_instances = fs.transform(train_instances)
    test_instances = fs.transform(test_instances)

    #Normalisation
    if scaler!=None:
        logging.info("Normalisation... ")
        scaler.fit_transform(train_instances)
        test_instances = scaler.transform(test_instances)

    classifiers = quality_models(candidate_families(), train_instances, judgements)
    print(classifiers)

    classifier = best_model(candidate_families(), train_instances, judgements)

    #build the best model
    log_info('Building model... ')
    classifier.fit(train_instances, judgements)

    log_info('Making predictions... ')
    decisions = classifier.predict(test_instances)    
    decisions_formatted = np.append(np.array('Label'), decisions)
    ids = ['ImageId'] + list(range(1, len(decisions_formatted)))
    output = np.column_stack((ids, decisions_formatted))
    pd.DataFrame(output).to_csv(output_path, header=False, index=False)

def main():
    init_logging('./logs/', current_timestamp()+'.log')
    run(MinMaxScaler(), 'data/results-scaling-minmax.csv')
    run(StandardScaler(), 'data/results-scaling-std.csv')
    run(None, 'data/results-no-scaling.csv')    

if __name__=='__main__':
    main()