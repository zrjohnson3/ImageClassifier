import os # for file handling
import pickle # for saving the classifier

from skimage.io import imread # for reading images
from skimage.transform import resize # for resizing images
import numpy as np # for numerical operations
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split # for splitting the data
from sklearn.model_selection import GridSearchCV # for hyperparameter tuning
from sklearn.svm import SVC # for support vector classifier

# prepare data
input_dir = '/Users/zachjohnson/PyCharm/MachineLearning/ImageClassifier_py/clf-data'
categories = ['empty', 'not_empty']

data = []
labels = []
# For each category
for category_index, category in enumerate(categories):
    # For each image in the category
    for file in os.listdir(os.path.join(input_dir, category)):
        # Read the image
        img_path = os.path.join(input_dir, category, file)
        img = imread(img_path)
        # Resize the image
        img = resize(img, (15, 15))
        # Add the image to the data
        data.append(img)
        # Add the label to the labels
        labels.append(category_index)

data = np.asarray(data)
labels = np.asarray(labels)
# print(data)
# print(labels)


# Split the data into a training set and a test set
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle= True, stratify=labels)

# train classifier Create a support vector classifier
# (A support vector classifier is a type of support vector
# machine that uses hyperplanes to separate data into classes)
classifier = SVC()

# Define the hyperparameters to tune (gamma is the kernel coefficient and C is the regularization parameter)
# This trains the classifier with all possible combinations of the hyperparameters
# This will be 12 combinations of hyperparameters because there are 3 values for gamma and 4 values for C
parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]

# Tune the hyperparameters
# (This will take a while)
grid_search = GridSearchCV(classifier, parameters)

# grid_search.fit(x_train, y_train)
grid_search.fit(x_train.reshape(x_train.shape[0], -1), y_train)

# test performance
best_estimator = grid_search.best_estimator_

# y_prediction = best_estimator.predict(x_test)
y_prediction = best_estimator.predict(x_test.reshape(x_test.shape[0], -1))


score = accuracy_score(y_prediction, y_test)

print('{}% of samples were correctly classified'.format(score * 100))

print('Best hyperparameters:', grid_search.best_params_)
print('Best estimator:', best_estimator)
print('Test set score:', best_estimator.score(x_test.reshape(x_test.shape[0], -1), y_test))


pickle.dump(best_estimator, open('./clf-data/model.p', 'wb'))