import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split


class Perceptron:
    """Perceptron classifier.

    Attributes:
        learning_rate : float, learning speed (between 0.1 and 1.0)
        n_iter : integer, number of iterations
        random_state : seed for random number generator
        w_ : ndarray, weights after fitting
        errors_ : list, unsuccessful classifications
    """
    def __init__(self, learning_rate=0.01, n_iter=50, random_state=1):
        """Init.

        Parameters:
            learning_rate : float, between 0.1 and 1.0
            n_iter : integer, number of iterations
            random_state : seed for random number generator
        """
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state
 

    def fit(self, X, y):
        """Fits perceptron on the training data.

        Parameters:
        X : ndarray, training vector
        y : ndarray, target values

        Returns:
        self : class, fitted perceptron classifier
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.learning_rate * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    
    def net_input(self, X):
        """Calculates net input."""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    
    def predict(self, X):
        """"Returns class labels after unit step."""
        return np.where(self.net_input(X) >= 0.0, 1, -1)


def one_versus_rest(X_train, X_test, y_train, y_test):
    """Applies One Versus All method to a prepared dataset.

    Parameters:
        X_train : ndarray, part of dataset for training
        X_test : ndarray, part of dataset for testing
        y_train : ndarray, target values for training
        y_test : ndarray, target values for testing

    Returns:
        predicted : ndarray, labels predicted for a given input vector
        errors : ndarray, number of errors in each epoch for all classifications
    """
    unique_classes = np.unique(y_train)
    scores = np.zeros((len(unique_classes), X_test.shape[0]))
    errors = []
    for i in range(len(unique_classes)):
        # changing target to 1 for class to classify and -1 for other classes
        y_encoded = np.where(y_train == unique_classes[i], 1, -1)
        # create a new perceptron classifier for each class
        classifier = Perceptron(learning_rate=0.0005, n_iter=100)
        classifier.fit(X_train, y_encoded)
        errors.append(classifier.errors_)
        # get scores for test vectors for the chosen class classification
        scores[i] = classifier.net_input(X_test)
    # choose the highest score for each element and get it's corresponding class
    predicted = [unique_classes[score] for score in np.argmax(scores, 0)]
    accuracy = sum(predicted==y_test)/len(y_test)
    print(f'Obtained accuracy: {accuracy:.2f}')
    return predicted, np.asarray(errors)


# import Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
labels = iris.target_names
# data normalization to 0-1 range
for i in range(X.shape[1]):
    X[:,i] = (X[:,i]-np.amin(X[:,i]))/(np.amax(X[:,i])-np.amin(X[:,i]))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# multiclass classification with One Versus All method
predictions, errors = one_versus_rest(X_train, X_test, y_train, y_test)
predicted_names = [labels[prediction] for prediction in predictions]
# plot errors for each of performed classifications
for i in range(len(errors)):
    plt.plot(range(errors.shape[1]), errors[i, :], marker='.', label=labels[i])
plt.plot(range(errors.shape[1]), np.sum(errors, 0), label='sum')
plt.legend(title='Class name:')
plt.title('Training errors')
plt.xlabel('epoch')
plt.ylabel('errors')
plt.show()
