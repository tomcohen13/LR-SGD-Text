import math
import numpy as np
from scipy.io import loadmat
from scipy.sparse import vstack
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle

# Load in data
news = loadmat('news_crypt_space.mat')

# Aggregate training data by stacking [crypt, space] sparse matrices, and converting to np array
train_data = vstack((news['crypt'], news['space'])).tocsr().toarray()
test_data = vstack((news['crypt_test'], news['space_test'])).tocsr().toarray()

# Create training labels (-1 for crypt, +1 for space)
train_labels = np.concatenate((np.full((news['crypt'].shape[0]), -1), np.ones((news['space'].shape[0]))))
test_labels = np.concatenate((np.full((news['crypt_test'].shape[0]), -1), np.ones((news['space_test'].shape[0]))))

# [Instead of hill-climbing, we do stochastic gradient descent.]

# Some constants
train_len = train_data.shape[0]     # 1187
test_len = test_data.shape[0]       # 787
d = train_data.shape[1]             # 61188
eta = 0.25                          # 1/4
passes = 9                          # w(0), w(1), ... w(8) 9 total weights

# Sigmoid function utility
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Wrap process in callable function
def stochastic_gradient_descent(random_state):
    # Initialize weights to 0
    # Have a continuously changing weight vector "w", 
    # and also store w values at the end of each pass-through in "weights"
    w = np.zeros(d)
    weights = np.zeros((passes, d))

    # Calculate objective value (log loss), train and test error rate in each iteration
    objective_values = np.zeros(passes)
    train_error_rates = np.zeros(passes)
    test_error_rates = np.zeros(passes)

    # Indices for shuffling
    indices = np.arange(train_len)
    np.random.seed(random_state)

    # Make 8 passes through training data
    for p in range(1, passes):
        print('---------------------------------------')
        print('Pass {}'.format(p))

        # Shuffle order of training data with random seed
        np.random.shuffle(indices)
        X = train_data[indices]
        y = train_labels[indices]

        # Iterate through training examples in this random order
        for i in range(train_len):
            # Compute the gradient
            grad = (1 - sigmoid(y[i] * np.dot(X[i], w))) * y[i] * X[i]
            # Update weights
            w = w + eta * grad

        # Save and print weights after each pass-through
        weights[p] = w
        print('Weights ({}): {}'.format(p, weights[p]))

        # Calculate objective function values
        objective_values[p] = np.log(1 + np.exp(-1 * (train_labels * np.dot(train_data, weights[p])))).sum() / train_len

        print('Objective value ({}): {}'.format(p, objective_values[p]))

        # Make predictions and calculate error rate
        train_predictions = np.where(np.dot(train_data, weights[p]) > 0, 1, -1)
        train_error_rates[p] = np.where(train_predictions != train_labels, 1, 0).sum() / train_len

        # Calculate test error rate as well
        test_predictions = np.where(np.dot(test_data, weights[p]) > 0, 1, -1)
        test_error_rates[p] = np.where(test_predictions != test_labels, 1, 0).sum() / test_len

        print('Train error rate ({}): {}'.format(p, train_error_rates[p]))
        print('Test error rate ({}): {}'.format(p, test_error_rates[p]))

    return objective_values, train_error_rates, test_error_rates

# Run with random seed 0
objective_values, train_error_rates, test_error_rates = stochastic_gradient_descent(0)

# Plot log losses as a function of p.
plt.figure()
plt.plot(range(1, passes), objective_values[1:])
plt.xlabel(r"$p$")
plt.ylabel(r"$f(w^{(p)})$")
plt.title(r"Objective value over pass-throughs")
plt.show()

# Plot train and test error rates as a function of p.
plt.figure()
plt.plot(range(1, passes), train_error_rates[1:], label = 'train error rate')
plt.plot(range(1, passes), test_error_rates[1:], label = 'test error rate')
plt.xlabel(r"$p$")
plt.ylabel("Error rate")
plt.legend()
plt.title(r"Train and test error rates of classifiers corresponding to $w^{(p)}$ as a function of $p$")
plt.show()

# Run 10 times with 10 different random seeds
k = 10

f = np.zeros((k, passes))
tr = np.zeros((k, passes))
te = np.zeros((k, passes))

for i in range(k):
    print('---------------------------------------')
    print('Random seed {}'.format(i))
    f[i], tr[i], te[i] = stochastic_gradient_descent(i)

# Calculate means across these executions
f_means = np.mean(f, axis=0)
tr_means = np.mean(tr, axis=0)
te_means = np.mean(te, axis=0)

# Calculate standard deviations across these executions
f_stds = np.std(f, axis=0)
tr_stds = np.std(tr, axis=0)
te_stds = np.std(te, axis=0)

# Plot!

# Log losses:
plt.figure()
plt.errorbar(range(1, passes), f_means[1:], yerr=f_stds[1:], capsize=5)
plt.xlabel(r"$p$")
plt.ylabel(r"$f(w^{(p)})$")
plt.title("Average objective values over {} iterations".format(k))
plt.show()

# Error rates:
plt.figure()
plt.errorbar(range(1, passes), tr_means[1:], yerr=tr_stds[1:], label='train error rate', capsize=5)
plt.errorbar(range(1, passes), te_means[1:], yerr=te_stds[1:], label='test error rate', capsize=5)
plt.xlabel(r"$p$")
plt.ylabel("Error rate")
plt.legend()
plt.title("Average train and test error rates over {} iterations".format(k))
plt.show()

# Report average train and test for w^{(8)}
print('Average train error rate for w({}): {}'.format(passes - 1, tr_means[passes - 1]))
print('Average test error rate for w({}): {}'.format(passes - 1, te_means[passes - 1]))

# Results:
# Average train error rate for w(8): 0.0
# Average test error rate for w(8): 0.04625158831003812