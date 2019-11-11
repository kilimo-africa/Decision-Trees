import pandas as pd
import numpy as np
from pprint import pprint
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from collections.abc import Mapping
import networkx as nx
import pylab
import matplotlib as plt

data_headers = ['engine', 'turbo', 'weight', 'fueleco', 'fast']
dataset = pd.read_csv('id3_data.csv', names=data_headers)


def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy = np.sum([(-counts[i] / np.sum(counts)) *
                      np.log2(counts[i] / np.sum(counts))
                      for i in range(len(elements))])
    return entropy


def InformationGain(data, split_attribute_name, target_name="class"):
    total_entropy = entropy(data[target_name])
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    weighted_entropy = np.sum([(counts[i] / np.sum(counts)) *
                               entropy(data.where(data[split_attribute_name] == vals[i])
                                       .dropna()[target_name]) for i in range(len(vals))])
    information_gain = total_entropy - weighted_entropy
    return information_gain


def ID3(data, originaldata, features, target_attribute_name, parent_node_class=None):
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    elif len(data) == 0:
        return np.unique(originaldata[target_attribute_name])[
            np.argmax(np.unique(originaldata[target_attribute_name], return_counts=True)[1])
        ]
    elif len(features) == 0:
        return parent_node_class
    else:

        parent_node_class = np.unique(data[target_attribute_name])[
            np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])
        ]
        item_values = [InformationGain(data, feature, target_attribute_name)
                       for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        tree = {best_feature: {}}
        features = [i for i in features if i != best_feature]
        for value in np.unique(data[best_feature]):
            value = value
            sub_data = data.where(data[best_feature] == value).dropna()
            sub_tree = ID3(sub_data, dataset, features, target_attribute_name, parent_node_class)
            tree[best_feature][value] = sub_tree
        return tree


def predict(query, tree, default=1):
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]]
            except:
                return default
            result = tree[key][query[key]]
            if isinstance(result, dict):
                return predict(query, result)
            else:
                return result


def train_test_split(dataset):
    training_data = dataset.iloc[:15].reset_index(drop=True)
    testing_data = dataset.iloc[15:].reset_index(drop=True)
    return training_data, testing_data


training_data = train_test_split(dataset)[0]
testing_data = train_test_split(dataset)[1]


def test(data, tree):
    predicted_array = np.array([])
    queries = data.iloc[:, :-1].to_dict(orient="records")
    predicted = pd.DataFrame(columns=["predicted"])
    for i in range(len(data)):
        predicted.loc[i, "predicted"] = predict(queries[i], tree, 1)
        state = predicted.loc[i, "predicted"]
        pprint(state)
        if state == "yes":
            predicted_array = np.append(predicted_array, 1)
        else:
            predicted_array = np.append(predicted_array, 0)
    return predicted_array


tree = ID3(training_data, training_data, training_data.columns[:-1], 'fast')
pprint(tree)
arr = test(testing_data, tree)

def print_tree():
    G = nx.DiGraph(tree)
    q = list(tree.items())
    print(q)

    while q:
        v, d = q.pop()
        for nv, nd in d.items():
            G.add_edge(v, nv)
            if isinstance(nd, Mapping):
                q.append((nv, nd))
            np.random.seed(1)
    nx.draw(G, with_labels=True)
    pylab.show()

predicted_list = arr.tolist()
pprint(predicted_list)
expected = [1, 1, 0, 0, 1, 1, 1, 0]
results = confusion_matrix(expected, predicted_list)
print('Confusion Matrix :')
print(results)
print('Accuracy Score :', accuracy_score(expected, predicted_list))
print('Report : ')
print(classification_report(expected, predicted_list))

print_tree()

