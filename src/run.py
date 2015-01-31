from sklearn.ensemble import RandomForestClassifier
import csv_io

def main():
	#Read in the training data and train the model
    train_data = csv_io.read_csv("data/train.csv")
    #the first column of the training set will be the judgements
    judgements = [str(int (x[0])) for x in train_data]
    train_instances = [x[1:] for x in train_data]

    #train the model
    classifier = RandomForestClassifier(n_estimators=100)
    classifier.fit(train_instances, judgements)

    #Read the test data and make predictions
    test_data = csv_io.read_csv("data/test.csv")
    decisions = classifier.predict(test_data)
    formatted_decisions = [["ImageId", "Label"]]

    count = 1
    for decision in decisions:
    	formatted_decisions.append([str(count), decision])
    	count += 1

    #write to a results CSV file 
    csv_io.write_csv("data/results.csv", formatted_decisions)

if __name__=="__main__":
    main()