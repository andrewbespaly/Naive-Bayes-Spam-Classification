import csv
import numpy as np

# Function for testing
# Count how many spam emails are in the set
def count_spam(dataset, dataset_name):
    spam = 0
    notspam = 0
    for i in range(0, len(dataset)):
        if dataset[i][-1] == 1:
            spam = spam + 1
        else:
            notspam = notspam + 1
    print()
    print('In', dataset_name, ':')
    print('spam:', spam)
    print('notspam:', notspam)
    print()

# Calculate the prior for each class in the dataset
def calc_prior(dataset):
    spam_count = 0
    for i in range(0, len(dataset)):
        if dataset[i][-1] == 1:
            spam_count = spam_count+1
    spam_prior = spam_count / len(dataset)
    nonspam_prior = (len(dataset)-spam_count) / len(dataset)
    return spam_prior, nonspam_prior


# Given a class, calculate the mean and standard deviation and return it in a 2D matrix([mean1,stddev1],[mean2, stddev2],...]
def calc_mean_stddev(class_data):
    curr_mean_list = np.mean(class_data, axis=0)
    curr_stddev_list = np.std(class_data, axis=0)

    for i in range(0, len(curr_stddev_list)):
        if(curr_stddev_list[i] == 0):
            curr_stddev_list[i] = 0.0001

    curr_mean_list = np.delete(curr_mean_list, -1)
    curr_stddev_list = np.delete(curr_stddev_list, -1)
    return curr_mean_list, curr_stddev_list

# Separate the two classes in the dataset
def separate_spam(dataset):
    spam_set= []
    nonspam_set = []
    for i in range(0, len(dataset)):
        if dataset[i][-1] == 1:
            spam_set.append(dataset[i])
        else:
            nonspam_set.append(dataset[i])
    return spam_set, nonspam_set

# Print the confusion matrix
def create_matrix(list):
    print('Pred|Act Spam\tNonSpam')
    print('Spam\t', list[0], '\t', list[1])
    print('NonSpam\t', list[2], '\t', list[3])

# Calculates the probabilities of each class and determines which has the larger value
# Return 1 for spam, 2 for nonspam
def classifier(data_to_classify, dataset):
    spam_set, nonspam_set = separate_spam(dataset)
    spam_mean, spam_std = calc_mean_stddev(spam_set)
    nonspam_mean, nonspam_std = calc_mean_stddev(nonspam_set)
    spam_prior, nonspam_prior = calc_prior(dataset)

    # Loop through all attributes and calculate probabilities
    spam_prob_list = []
    for feature in range(0, len(spam_set[0])-1):
        spam_prob_list.append(math_calculation(data_to_classify[feature], spam_mean[feature], spam_std[feature]))

    nonspam_prob_list = []
    for feature in range(0, len(nonspam_set[0])-1):
        nonspam_prob_list.append(math_calculation(data_to_classify[feature], nonspam_mean[feature], nonspam_std[feature]))

    # log each result to prevent underflow from occuring
    # first check if item is 0, if it is, make it a very small number
    if(spam_prior == 0):
        spam_prior = 10**-300
    total_spam_prob = np.log10(spam_prior)

    for item in range(0, len(spam_prob_list)):
        if(spam_prob_list[item]==0):
            spam_prob_list[item] = 10**-300
        total_spam_prob += np.log10(spam_prob_list[item])

    total_nonspam_prob = np.log10(nonspam_prior)
    for item in range(0, len(nonspam_prob_list)):
        if (nonspam_prob_list[item] == 0):
            nonspam_prob_list[item] = 10**-300
        total_nonspam_prob += np.log10(nonspam_prob_list[item])

    if(total_spam_prob >= total_nonspam_prob):
        return 1
    else:
        return 0

# Math needed using new feature value, mean, and standard deviation
def math_calculation(value, mean, std):
    first_value = (1/(np.sqrt(2*np.pi)*std))
    upper_e_value = (-((value-mean)**2))
    lower_e_value = 2*((std)**2)
    all_e_value = upper_e_value / lower_e_value
    second_value = np.exp(all_e_value)
    final_value = first_value*second_value
    return final_value

# Read the training data and put it into data variable
data = []
file = open(r'spambase.data')
reader = csv.reader(file)
next(reader, None)
for row in reader:
    floatRow = []
    for item in row:
        floatRow.append(float(item))
    data.append(floatRow)
file.close()

# Fill the training and test datasets with equal amounts of both spam and not spam
training_data = []
test_data = []
for i in range(0, 906):
    training_data.append(data[i])
for i in range(906, 1812):
    test_data.append(data[i])
for i in range(1812, 3206):
    training_data.append(data[i])
for i in range(3206, 4600):
    test_data.append(data[i])

test_correct = 0

confusion_matrix = np.zeros((4))  # 0: spam-spam, 1: spam-nonspam, 2: nonspam-spam, 3:nonspam-nonspam, --actual-predicted
for row in range(0, len(test_data)):
    prediction = classifier(test_data[row], training_data)
    if(prediction == test_data[row][-1]):
        test_correct += 1
        if(prediction == 1):
            confusion_matrix[0] += 1
        else:
            confusion_matrix[3] += 1
    else:
        if(prediction == 1):
            confusion_matrix[2] += 1
        else:
            confusion_matrix[1] += 1

create_matrix(confusion_matrix)

print()
print(test_correct, '/', len(test_data))
test_accuracy = (test_correct / len(test_data)) * 100
print(test_accuracy)
