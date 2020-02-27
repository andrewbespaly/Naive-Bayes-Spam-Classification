import csv
import numpy as np

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


def separate_spam(dataset):
    spam_set= []
    nonspam_set = []
    for i in range(0, len(dataset)):
        if dataset[i][-1] == 1:
            spam_set.append(dataset[i])
        else:
            nonspam_set.append(dataset[i])
    return spam_set, nonspam_set


# def classifier(data_to_classify, dataset):
#     spam_set, nonspam_set = separate_spam(dataset)
#     spam_mean, spam_std = calc_mean_stddev(spam_set)
#     nonspam_mean, nonspam_std = calc_mean_stddev(nonspam_set)
#     spam_prior, nonspam_prior = calc_prior(dataset)
#
#     # Loop through all attributes and calculate probabilities
#     for feature in range(0, len(dataset[0])-1):


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
file = open(r'C:\Users\Andrew\Desktop\spambase\spambase.data')
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

# count_spam(training_data, 'Training Data')
# # count_spam(test_data, 'Test Data')

# Make 2 Separate Classes, 1 for spam, 1 for non-spam,
# then loop through all the attributes of each and calculate mean and stddev

train_spam, train_non_spam = separate_spam(training_data)

testset = [
    [3.0, 5.1, 1],
    [4.1, 6.3, 1],
    [7.2, 9.8, 1],
    [2.0, 1.1, 0],
    [4.1, 2.0, 0],
    [8.1, 9.4, 0]
]

test_set_pos = [
    [3.0, 5.1, 1],
    [4.1, 6.3, 1],
    [7.2, 9.8, 1]
]

test_set_neg = [
    [2.0, 1.1, 0],
    [4.1, 2.0, 0],
    [8.1, 9.4, 0]
]

testmeanpos, teststdpos = calc_mean_stddev(test_set_pos)
testmeanneg, teststdneg = calc_mean_stddev(test_set_neg)

# print(testmeanpos)
# print(teststdpos)
# print()
# print(testmeanneg)
# print(teststdneg)
# testspamprior, testnonspamprior = calc_prior(testset)
# print(testspamprior, testnonspamprior)

num = math_calculation(6.3, 4.2, 3.7)
print(num)
