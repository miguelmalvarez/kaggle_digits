from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import pandas as pd
import numpy as np
import logging
import time
import datetime
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

def accuracy(decisions, judgements):
    accuracy = 0.0
    for (decision, judgement) in zip(decisions, judgements):
        if (decision == judgement):
            accuracy += 1
    if (accuracy != 0):        
        accuracy = accuracy/len(decisions)
    return accuracy

def log_info(message):
    ts = time.time()
    logging.info(message + datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))

def init_logging():
    logging.basicConfig(format='%(message)s', level=logging.INFO, filename='history.log')

def candidates():
    classifiers = []
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    classifiers.append(["RandomForest-100 ", rf])
    classifiers.append(["Naive-Bayes", GaussianNB()])
    classifiers.append(["LinearSVC", LinearSVC()])
    return classifiers

def xval(classifier, train_instances, judgements):
    log_info('Crossvalidation started... ')    
    cv = cross_validation.StratifiedKFold(np.array(judgements), n_folds=5)

    avg_quality = 0.0
    for train_index, test_index in cv:        
        train_cv, test_cv = train_instances[train_index], train_instances[test_index]
        train_judgements_cv, test_judgements_cv = judgements[train_index], judgements[test_index]
        decisions_cv = classifier.fit(train_cv, train_judgements_cv).predict(test_cv)
        quality = accuracy(decisions_cv, test_judgements_cv)
        avg_quality += quality
        log_info('Quality of split... ' + str(quality))
    quality = avg_quality/len(cv)
    log_info('Estimated quality of model... ' + str(quality))

    return quality

def best_model(classifiers, train_instances, judgements):
    best_quality = 0.0
    best_classifier = None

    for name, classifier in classifiers:
        log_info('Considering classifier... ' + name)
        quality = xval(classifier, train_instances, judgements)
        if (quality > best_quality):
            best_quality = quality
            best_classifier = [name, classifier]

    log_info('Best classifier... ' + best_classifier[0])

    return best_classifier[1]

def main():
    init_logging()
    log_info('============== \nClassification started... ')

    log_info('Reading training data... ')
    train_data = pd.read_csv('data/train.csv', header=0).values
    #the first column of the training set will be the judgements
    judgements = np.array([str(int (x[0])) for x in train_data])
    train_instances = np.array([x[1:] for x in train_data])

    classifier = best_model(candidates(), train_instances, judgements)

    #build the best model
    log_info('Building model... ')
    classifier.fit(train_instances, judgements)

    log_info('Reading testing data... ')
    test_data =  pd.read_csv('data/test.csv', header=0).values
    decisions = classifier.predict(test_data)

    log_info('Output results... ')
    decisions_formatted = np.append(np.array('Label'), decisions)
    ids = ['ImageId'] + list(range(1, len(decisions_formatted)))
    output = np.column_stack((ids, decisions_formatted))
    pd.DataFrame(output).to_csv('data/results.csv', header=False, index=False)

if __name__=='__main__':
    main()