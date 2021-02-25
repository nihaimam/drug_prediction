import re
import time  # to keep track of time
from sklearn.decomposition import TruncatedSVD
from imblearn.over_sampling import SVMSMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier

start_time = time.time()
train = "train.dat"
test = "test.dat"

# we have 100000 features in our training set
# added 1 one more because i was getting an error
num_features = 100001

def get_data(filename, filetype):
    with open(filename, "r") as in_file:
        input_lines = in_file.readlines()
    if filetype == "train":
        # training data has labels
        # create a list of 800 labels
        labels = [int(l[0]) for l in input_lines]
        for i, item in enumerate(labels):
            if (item == 0):
                labels[i] = -1
        docs = [re.sub(r'[^\w]', ' ', l[1:]).split() for l in input_lines]
    else:
        # test data has no labels
        labels = []
        docs = [re.sub(r'[^\w]', ' ', l).split() for l in input_lines]

    # use docs to add features
    features = []

    for d in docs:
        line = [0] * num_features
        for i, val in enumerate(d):
            line[int(val)] = 1
        features.append(line)

    return features, labels


# load train data and separate the labels and features
features, labels = get_data(train, "train")

# load test data and separate the labels and features
t_features, t_labels = get_data(test, "test")

# reduce dimensionality since we have 100000 features
# according to google truncatedSVD is better than PCA for sparse
tsvd = TruncatedSVD(algorithm='randomized', n_components=1200, n_iter=55, random_state=22)
tsvd_fit = tsvd.fit(features, labels)  # killer big O
less_features = tsvd_fit.transform(features)

# since the data-set imbalanced stack overflow recommended using SMOTE
svmsmote = SVMSMOTE(random_state=42)
# using fit_sample to re-sample the data-set
less_features, labels = svmsmote.fit_resample(less_features, labels)

# reduce the test features too
less_test_features = tsvd_fit.transform(t_features)

# start classifying
classifiers_used = ["Perceptron",
                    "Random Forest",
                    "Decision Tree",
                    "Stochastic Gradient Descent",
                    "Controlled Stochastic Gradient Descent"]
classifiers = [Perceptron(alpha=0.0001, class_weight={-1: 1, 1: 1.5}, random_state=57),
               RandomForestClassifier(class_weight={-1: 1, 1: 1.5}, random_state=55),
               DecisionTreeClassifier(class_weight={-1: 1, 1: 1.5}, random_state=53),
               SGDClassifier(),
               SGDClassifier(alpha=0.0001, class_weight={-1: 1, 1: 1.5}, random_state=51)]

i = 0

# loop through all the classifiers and apply it to the data
for name, clf in zip(classifiers_used, classifiers): #i in range(0, len(classifiers_used)):
    print("Using Classifier: " + name)

    # fit and predict labels
    clf.fit(less_features, labels)
    pred = clf.predict(less_test_features)

    if i == 0:
        out_file = 'perceptron.dat'
    elif i == 1:
        out_file = 'decisiontree.dat'
    elif i == 2:
        out_file = 'randomforest.dat'
    elif i == 3:
        out_file = 'sgd.dat'
    elif i == 4:
        out_file = 'controlledsgd.dat'
    else:
        out_file = 'out.dat'

    i += 1

    print('Labels in', out_file)

    output = open(out_file, 'w')
    for p in pred:
        if int(p) == -1:
            p = 0
        output.write(str(p))
        output.write("\n")
    output.close()

print("completion time  :  --- %s minutes ---" % ((time.time() - start_time)/60))