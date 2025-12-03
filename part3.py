import time
import os

from CS170Project2Part2 import (
    NearestNeighbor, Validator, load_dataset, normalize
)



def backward_elimination(data, classifier, validator):
    features = len(data[0]) - 1  # Exclude class label
    curr_set = list(range(1, features + 1))
    trace = []

    full_acc = validator.leave_one_out(data, curr_set)

    print(f"Running nearest neighbor with all features (default rate), using “leaving-one-out” evaluation, I get an accuracy of {full_acc*100:.1f}%\n")
    print("Beginning search.\n")


    best_overall = full_acc
    best_score = curr_set.copy()



    while len(curr_set) > 1:
        feature_to_remove = None
        best_so_far = -1

        for feature in curr_set:
            temp_set = [x for x in curr_set if x != feature]
            score = validator.leave_one_out(data, temp_set)
            trace.append(f"Using features {temp_set} accuracy is {score*100:.1f}%")
            if score > best_so_far:
                best_so_far = score
                feature_to_remove = feature
    
        curr_set.remove(feature_to_remove)
        trace.append(f"\nFeature set {curr_set} was best, accuracy is {best_so_far*100:.1f}%\n")

        if best_so_far < best_overall:
            trace.append("(Warning, Accuracy has decreased!)")

        if best_so_far > best_overall: #update best overall and best score for next iteration
            best_overall = best_so_far
            best_score = curr_set.copy()

    return best_score, best_overall,  trace

def forward_selection(data, classifier, validator):
    tot_features = len(data[0]) - 1  # Exclude class label
    current_set = []
    trace = []

    # Score with no features testing empty set
    base_acc =validator.leave_one_out(data,current_set)
    print(f"Running nearest neighbor with no features (default rate), using “leaving-one-out” evaluation, I get an accuracy of {base_acc*100:.1f}%")
    print("Beginning search.")

    best_subset = current_set.copy()   
    best_score = base_acc    

    # Add one feature at each level of the search tree
    for level in range(1, tot_features + 1):

        feature_to_add = None
        level_best_accuracy = -1

        # Test each feature not in current_set
        for feature in range(1, tot_features + 1):
            if feature not in current_set:
                trial = current_set + [feature]
                acc = validator.leave_one_out(data, trial)
                trace.append(f"Using feature(s) {trial} accuracy is {acc*100:.1f}%")

                if acc > level_best_accuracy:
                    level_best_accuracy = acc
                    feature_to_add = feature

        # Add the best feature found at each level of the tree
        if feature_to_add is not None:
            new_set = current_set + [feature_to_add]

            if level_best_accuracy < best_score:
                trace.append("(Warning, Accuracy has decreased!)")

            print(f"Feature set {new_set} was best, accuracy is {level_best_accuracy*100:.1f}%")

            current_set = new_set

            # Store the best subset
            if level_best_accuracy > best_score:
                best_score = level_best_accuracy
                best_subset = new_set.copy()


    return  best_subset, best_score, trace

def main():
    print("Welcome to Your Name's Feature Selection Algorithm.\n")

    filename = input("Type in the name of the file to test : ")

    print("\nType the number of the algorithm you want to run.\n")
    print("1) Forward Selection")
    print("2) Backward Elimination")
    print("3) Special Algorithm\n")

    choice = input()

    data = load_dataset(filename)
    print(
        f"\nThis dataset has {len(data[0]) - 1} features "
        f"(not including the class attribute), with {len(data)} instances."
    )

    print("Please wait while I normalize the data...", end="")
    data = normalize(data)
    print(" Done!\n")

    classifier = NearestNeighbor()
    validator = Validator(classifier)

    if choice == "1":
        best_set, best_acc, trace = forward_selection(data, classifier, validator)
        for line in trace:
            print(line)
        print(f"Finished search!! The best feature subset is {best_set}, "
              f"which has an accuracy of {best_acc*100:.1f}%")

    elif choice == "2":
        best_set, best_acc, trace = backward_elimination(data, classifier, validator)
        for line in trace:
            print(line)
        print(f"Finished search!! The best feature subset is {best_set}, "
              f"which has an accuracy of {best_acc*100:.1f}%")

    elif choice == "3":
        print("\nSpecial Algorithm not implemented.\n")

    else:
        print("\nInvalid choice.\n")

main()