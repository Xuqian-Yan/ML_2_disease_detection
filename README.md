# The original description from https://www.kaggle.com/c/ml-project-2

# Description

In this project you will classify a person's cognitive health status only from an MR scan of their brain.

After gaining first experience with MRI data during the first project, we want to tackle a more difficult task in the second project. The problem is to diagnose a person's cognitive health status, learning from a set of expert-labeled training images. These labels are typically acquired by performing several time-consuming neuro-psychological tests. Replacing these tests with an automatic analysis of MRI scans would save a considerable amount of resources.

The MRI scans are labeled by the severity of cognitive decline: healthy, very mild degeneration, mild degeneration, moderate degeneration.

# Evaluation

There are four potential classes (healthy, very mild degeneration, mild degeneration, moderate degeneration) in our problem. A prediction specifies the probability of a sample to be in each of those classes.

The evaluation metric for this competition is the mean Spearman's rank correlation coefficient between your prediction and the true class probabilities. Spearman's rank correlation score is only sensitive to the order of class probabilities in terms of their absolute value. It is, however, not sensitive to the exact values of the class probabilities. All that matters is the rank order of the class probabilities.

The Spearman score used in this competition is equal to the function scipy.stats.spearmanr

For instance, let us assume the true label is (0.823, 0.149, 0.027, 0.001), i.e. the probability of the first class (healthy) is 82.3%, etc. The following two predictions would both achieve the maximum score of 1.0:

(0.60, 0.30, 0.06, 0.04) and (0.70, 0.15, 0.10, 0.05)

# Submission Format

For every test sample, submission files should contain two columns: ID and Prediction. Prediction should be a space-delimited list containing the class probabilities.

The file should contain a header and have the following format:

ID,Prediction

1,0.60 0.30 0.06 0.04

# Data

X_train.npy - the training set provided as numpy array with shape (290, 6443008). The rows run over samples, the columns over features. Basically, each feature is a different voxel (3D pixel) of the image. You can get the 3D structure back with numpy. reshape(X_train, (-1, 176, 208, 176)).

X_test.npy - the test set provided in the same format as the training set, but with shape (126, 6443008).

train_labels.csv - labels for the training samples in X_train.npy. Row k in train_labels.csv contains the class probabilities for the sample in row k of the numpy array X_train.npy.

# My final score

Mean Spearman's rank correlation coefficient = 0.85454

Ranked 32%

