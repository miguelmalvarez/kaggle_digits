import pandas as pd
import numpy as np
import logging
import time
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV

# Formated current timestamp
def current_timestamp():
    ts = time.time()
    return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

# Log message with timestamp
def log_info(message):
    ts = time.time()
    logging.info(message + " " + current_timestamp())

# Initialise logging
def init_logging(log_file_path):
    logging.basicConfig(format='%(message)s', level=logging.INFO, filename=log_file_path)

# List of candidate family classifiers with parameters for grid search
# [name, classifier object, parameters].
def candidate_families():
    candidates = []
    svm_tuned_parameters = [{'kernel': ['poly'], 'degree': [1, 2, 3, 4]}]
    candidates.append(["SVM", SVC(C=1), svm_tuned_parameters])
    rf_tuned_parameters = [{"n_estimators": [250, 500, 1000]}]
    candidates.append(["RandomForest", RandomForestClassifier(n_jobs=-1), rf_tuned_parameters])        
    knn_tuned_parameters = [{"n_neighbors": [1, 3, 5, 10, 20]}]
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

# Returns the best model from a set of model families given  training data using crosvalidation
def best_model(classifier_families, train_instances, judgements):
    best_quality = 0.0
    best_classifier = None    
    classifiers = []
    for name, model, parameters in classifier_families:
        classifiers.append(best_config([name, model], parameters, train_instances, judgements))

    for name, quality, classifier in classifiers:
        log_info('Considering classifier... ' + name)
        if (quality > best_quality):
            best_quality = quality
            best_classifier = [name, classifier]

    log_info('Best classifier... ' + best_classifier[0] + "-" + str(best_classifier[2]))
    return best_classifier[1]

# Return the best 
def best_config(model_info, parameters, train_instances, judgements):
    [name, model] = model_info
    log_info('Grid search for... ' + name)
    clf = GridSearchCV(model, parameters, cv=5, scoring="accuracy", verbose=5, n_jobs=4)
    clf.fit(train_instances, judgements)
    best_estimator = clf.best_estimator_
    log_info('Best configuration: ' + str(clf.best_params_) + str(clf.best_score_))
    return [str(clf.best_params_), clf.best_score_, best_estimator]

# Run the data and over multiple classifier and output the data in a csv file
def run(output_path):
    file_log_path = './history'+current_timestamp()+'.log'
    init_logging(file_log_path)
    log_info('============== \nClassification started... ')

    log_info('Reading training data... ')
    train_data = pd.read_csv('data/train.csv', header=0).values
    #the first column of the training set will be the judgements
    judgements = np.array([str(int (x[0])) for x in train_data])
    train_instances = np.array([x[1:] for x in train_data])
    train_instances = [[float(x) for x in instance] for instance in train_instances]

    log_info('Reading testing data... ')
    test_data = pd.read_csv('data/test.csv', header=0).values
    test_instances = np.array([x[0:] for x in test_data])
    test_instances = [[float(x) for x in instance] for instance in test_instances]
    
    #Feature selection
    logging.info("Selecting features... ")
    fs = feature_selection(train_instances)
    train_instances = fs.transform(train_instances)
    test_instances = fs.transform(test_instances)

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
    run('data/results.csv')

if __name__=='__main__':
    main()