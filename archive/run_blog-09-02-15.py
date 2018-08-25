from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
import pandas as pd
import logging
import time
import datetime
import numpy as np
 
def log_info(message):
    ts = time.time()
    logging.info(message + " " + datetime.datetime
        .fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))

def init_logging(log_file_path):
    logging.basicConfig(format='%(message)s', level=logging.INFO, filename=log_file_path)

def xval(classifier, train_instances, judgements):
    log_info('Crossvalidation started... ')    
    cv = cross_validation.StratifiedKFold(np.array(judgements), n_folds=5)

    avg_quality = 0.0
    for train_index, test_index in cv:        
        train_cv, test_cv = train_instances[train_index], train_instances[test_index]
        train_judgements_cv, test_judgements_cv = judgements[train_index], judgements[test_index]
        decisions_cv = classifier.fit(train_cv, train_judgements_cv).predict(test_cv)
        quality = accuracy_score(decisions_cv, test_judgements_cv)
        avg_quality += quality
        log_info('Quality of split... ' + str(quality))
    quality = avg_quality/len(cv)
    log_info('Estimated quality of model... ' + str(quality))

def main():
    init_logging('./history.log')
    log_info('============== \nReading training data... ')
    train_data = pd.read_csv('data/train-sample.csv', header=0).values
    #the first column of the training set will be the judgements
    judgements = np.array([str(int (x[0])) for x in train_data])
    train_instances = np.array([x[1:] for x in train_data])
 
    classifier = RandomForestClassifier(n_estimators=100)
    log_info('Cross-validation... ')
    quality = xval(classifier, train_instances, judgements)

    log_info('Building model... ')
    classifier.fit(train_instances, judgements)
 
    log_info('Reading testing data... ')
    test_data =  pd.read_csv('data/test-sample.csv', header=0).values
    decisions = classifier.predict(test_data)

    log_info('Output results... ')
    decisions_formatted = np.append(np.array('Label'), decisions)
    ids = ['ImageId'] + list(range(1, len(decisions_formatted)))
    output = np.column_stack((ids, decisions_formatted))
    pd.DataFrame(output).to_csv('data/results.csv', header=False, index=False)

if __name__=='__main__':
    main()