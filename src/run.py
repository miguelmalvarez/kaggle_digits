from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import pandas as pd
import numpy as np

def accuracy(decisions, judgements):
    accuracy = 0.0
    for (decision, judgement) in zip(decisions, judgements):
        if (decision == judgement):
            accuracy += 1
    if (accuracy != 0):        
        accuracy = accuracy/len(decisions)
    return accuracy

def main():
	#Read in the training data and train the model
    train_data = pd.read_csv('data/train.csv', header=0).values
    #the first column of the training set will be the judgements
    judgements = np.array([str(int (x[0])) for x in train_data])
    train_instances = np.array([x[1:] for x in train_data])

    classifier = RandomForestClassifier(n_estimators=100)

    # Crossvalidation
    cv = cross_validation.StratifiedKFold(np.array(judgements), n_folds=5)

    quality = 0.0
    for train_index, test_index in cv:        
        train_cv, test_cv = train_instances[train_index], train_instances[test_index]
        train_judgements_cv, test_judgements_cv = judgements[train_index], judgements[test_index]
        decisions_cv = classifier.fit(train_cv, train_judgements_cv).predict(test_cv)

        quality += accuracy(decisions_cv, test_judgements_cv)
    quality = quality/len(cv)
    print('Quality ' + str(quality))

    #train the model    
    classifier.fit(train_instances, judgements)

    #Read the test data and make predictions
    test_data =  pd.read_csv('data/test.csv', header=0).values
    decisions = classifier.predict(test_data)

    # Output results
    decisions_formatted = np.append(np.array('Label'), decisions)
    ids = ['ImageId'] + list(range(len(decisions_formatted) -1))
    output = np.column_stack((ids, decisions_formatted))
    pd.DataFrame(output).to_csv('data/results.csv', header=False, index=False)

if __name__=='__main__':
    main()