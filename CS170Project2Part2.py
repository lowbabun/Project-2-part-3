import math
import random


# Nearest Neighbor Classifier
class NearestNeighbor:
    def __init__(self):
        self.training_data = []

    def train(self, data):
        self.training_data = data

    def test(self, instance, feature_subset):
        
        best_distance = float('inf')
        best_label = None

        for other in self.training_data:

            # Compute Euclidean distance using only the selected features
            dist = 0.0
            for f in feature_subset:
                dist = dist + (instance[f] - other[f])**2
            dist = math.sqrt(dist)

            if dist < best_distance:
                best_distance = dist
                best_label = other[0] 

        return best_label



# Leave One Out Validator
class Validator:
    def __init__(self, classifier):
        self.classifier = classifier

    def leave_one_out(self, data, feature_subset):
        correct = 0

        for i in range(len(data)):
            # Leave instance i out for testing
            test_instance = data[i]

            # Train on all others instances
            train_data = data[:i] + data[i+1:]
            self.classifier.train(train_data)

            # Predict the class
            predicted = self.classifier.test(test_instance, feature_subset)
            actual = test_instance[0]

            if predicted == actual:
                correct = correct + 1

        return correct / len(data)


# Load the dataset
def load_dataset(filename):
    data = []
    with open(filename, "r") as f:
        for line in f:
            if line.strip() == "":
                continue
            nums = line.split()
            nums = [float(n) for n in nums]
            data.append(nums)
    return data



# Normalize features
def normalize(data):
    num_features = len(data[0])

    # Class column is excluded
    means = [0] * num_features
    stds = [0] * num_features

    # Compute the means
    for col in range(1, num_features):
        s = sum(row[col] for row in data)
        means[col] = s / len(data)

    # Compute the std deviations
    for col in range(1, num_features):
        variance = sum((row[col] - means[col])**2 for row in data) / len(data)
        stds[col] = math.sqrt(variance) if variance > 0 else 1

    # Normalize the data
    for row in data:
        for col in range(1, num_features):
            row[col] = (row[col] - means[col]) / stds[col]

    return data
