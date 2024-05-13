import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class NN:
    def __init__(self, csv):
        self.W1 = np.random.rand(10, 784) - 0.5
        self.b1 = np.random.rand(10, 1) - 0.5
        self.W2 = np.random.rand(10, 10) - 0.5
        self.b2 = np.random.rand(10, 1) - 0.5
        self.data = pd.read_csv(csv)
        self.data = np.array(self.data)
        np.random.shuffle(self.data)

        self.X_train, self.Y_train, self.X_dev, self.Y_dev = self.train_dev_split()

    def train_dev_split(self):
        rows, cols = self.data.shape     
        size = int(rows * 0.1)

        data_dev = self.data[0:size].T
        X_dev = data_dev[1:cols]
        X_dev = X_dev / 255
        Y_dev = data_dev[0]

        data_train = self.data[size:rows].T
        X_train = data_train[1:cols]
        X_train = X_train / 255
        Y_train = data_train[0]
        return X_train, Y_train, X_dev, Y_dev

    def ReLU(self, Z):
        return np.maximum(0,Z)

    def softmax(self, Z):
        return np.exp(Z) / sum(np.exp(Z))

    def one_hot(self, X):
        one_hot_X = np.zeros((X.size, X.max() + 1))
        one_hot_X[np.arange(X.size), X] = 1
        one_hot_X = one_hot_X.T
        return one_hot_X

    def ReLU_deriv(self, Z):
        return Z > 0

    def forward_prop(self, W1, b1, W2, b2, X):

        Z1 = W1.dot(X) + b1
        A1 = self.ReLU(Z1)
        Z2 = W2.dot(A1) + b2
        A2 = self.softmax(Z2)

        return Z1, A1, Z2, A2

    def backward_prop(self, Z1, A1, Z2, A2, W1, W2, X, Y):

        one_hot_Y = self.one_hot(Y)
        m = Y.size
        dZ2 = A2 - one_hot_Y
        dW2 = 1 / m * dZ2.dot(A1.T)
        db2 = 1 / m * np.sum(dZ2)
        dZ1 = W2.T.dot(dZ2) * self.ReLU_deriv(Z1)
        dW1 = 1 / m * dZ1.dot(X.T)
        db1 = 1 / m * np.sum(dZ1)
        return dW1, db1, dW2, db2

    def update_params(self, W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):

        W1 = W1 - alpha * dW1
        b1 = b1 - alpha * db1
        W2 = W2 - alpha * dW2
        b2 = b2 - alpha * db2

        return W1, b1, W2, b2

    def get_prediction(self, A2):
        return np.argmax(A2, 0)

    def get_accuracy(self, predictions, Y):
        return np.sum(predictions == Y) / Y.size

    def gradient_descent(self, X, Y, alpha, iterations):
        W1, b1, W2, b2 = self.W1, self.b1, self.W2, self.b2

        for i in range(iterations):
            Z1, A1, Z2, A2 = self.forward_prop(W1, b1, W2, b2, X)
            dW1, db1, dW2, db2 = self.backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
            W1, b1, W2, b2 = self.update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

            if i % 50 == 0:
                pred = self.get_prediction(A2)
                print(f"iteration: {i} | Accuracy: {self.get_accuracy(pred, Y)}") 
                
        return W1, b1, W2, b2

    def make_predictions(self, X, W1, b1, W2, b2):
        _, _, _, A2 = self.forward_prop(W1, b1, W2, b2, X)
        predictions = self.get_prediction(A2)
        return predictions

    def test_prediction(self, index, W1, b1, W2, b2):
        current_image = self.X_train[:, index, None]
        prediction = self.make_predictions(self.X_train[:, index, None], W1, b1, W2, b2)
        label = self.Y_train[index]
        print("Prediction: ", prediction)
        print("Label: ", label)
        
        current_image = current_image.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(current_image, interpolation='nearest')
        plt.show()

    def dev_accuracy(self, W1, b1, W2, b2):
        dev_predictions = self.make_predictions(self.X_dev, W1, b1, W2, b2)
        return self.get_accuracy(dev_predictions, self.Y_dev)
    
if __name__ == '__main__':
    nn = NN('digit-recognizer/train.csv')
    W1, b1, W2, b2 = nn.gradient_descent(nn.X_train, nn.Y_train, 0.10, 1000)
    print(nn.dev_accuracy(W1, b1, W2, b2))