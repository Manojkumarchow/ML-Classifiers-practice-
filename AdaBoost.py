# Note: This is just a basic AdaBoost Classifier done for my practice.
# Let's import the dependencies
# Dependencies:
	# sklearn
	# required datasets from sklearn package
print("<-----------AdaBoost Classifier----------->")
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets

# Load the dataset, here I'm going to use the iris dataset

iris = datasets.load_iris()
# Let's see the shape of our data
print("Data Shape: ", iris.data.shape)
print("Target Shape: ", iris.target.shape)
# split the dataset in order to validate our classifier's performance
print()
print("<------------------------------------------------------>")
print()
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4)

# Here I'm using 40% of the data for testing purpose

print("Training Data Shape: ", x_train.shape) #training data
print("Testing Shape: ", x_test.shape) #testing data
print()
print("<------------------------------------------------------>")
print()
# Here comes the classifier, we use the fit() to fit the data into our classifier


clf = AdaBoostClassifier(n_estimators=100).fit(x_train, y_train)

print("Score: ", clf.score(x_test, y_test)) 

# Thanks for having a look!