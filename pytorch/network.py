import time

import torch
import torch.nn as nn
import torch.optim as optim

'''
Problem set up: We have images in greyscale that have a pixel resolution of 28x28.
These can be a number from 0 to 9, this means K = 10 for the classification. 
Since our neural network (NN) expects a 1D vector, not a matrix, by flattening the inputs, we would get 28 x 28 = 784 features.

Input of first layer: 784 neurons, 1 per pixel.
Output before softmax: 10 scores.
Output after softmax: 10 numbers (probabilities) that sum to 1.

Based on d2l.ai reference, all probabilities must must be greater or equal to 0 and must sum to 1. 
Softmax allows us to transform logits into a probability distribution p_k = 1.
'''

class TorchNetwork(nn.Module):
    def __init__(self, sizes, epochs=10, learning_rate=0.01, random_state=1):
        super().__init__()

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.random_state = random_state

        torch.manual_seed(self.random_state)

        self.linear1 = nn.Linear(sizes[0], sizes[1])
        self.linear2 = nn.Linear(sizes[1], sizes[2])
        self.linear3 = nn.Linear(sizes[2], sizes[3])

        '''
        Our activations
        '''
        self.activation_func = torch.sigmoid
        self.output_func = torch.softmax

        '''
        Mean Square Loss
        '''
        self.loss_func = nn.MSELoss() #as specificed in the description

        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)

    '''
    Forward Pass
    Implementing the forward propagation algorithm to return the output of the network.
    '''
    def _forward_pass(self, x_train):
        #First hidden layer
        h1 = self.activation_func(self.linear1(x_train))
        #Second hidden layer
        h2 = self.activation_func(self.linear2(h1))
        #Output logits (raw scores for 10 classes)
        outlogit = self.linear3(h2)
        #Softmax to get them to 1, 1 per row (dim)
        probabilities = self.output_func(outlogit, dim=1)
        return probabilities

    '''
    Backward Pass
    Implementing the forward propagation algorithm to return the output of the network.
    '''

    def _backward_pass(self, y_train, output):
        #We use float instead of string, so they are not treated as integers, outputs are from the probabilities
        y_train = y_train.float()
        #Computing the MSE Loss
        loss = self.loss_func(output, y_train)
        #Computing all the gradients that are stored in each tensor
        loss.backward()
        #Stores the float from the tensor
        return loss.item()

    '''
    Weights
    Updating the network of weights per SDG.
    '''
    def _update_weights(self):
        #PyTorch loop looks at parameters' gradient and updates to change weights
        self.optimizer.step()

    def _flatten(self, x):
        return x.view(x.size(0), -1)       

    def _print_learning_progress(self, start_time, iteration, train_loader, val_loader):
        train_accuracy = self.compute_accuracy(train_loader)
        val_accuracy = self.compute_accuracy(val_loader)
        print(
            f'Epoch: {iteration + 1}, ' \
            f'Training Time: {time.time() - start_time:.2f}s, ' \
            f'Learning Rate: {self.optimizer.param_groups[0]["lr"]}, ' \
            f'Training Accuracy: {train_accuracy * 100:.2f}%, ' \
            f'Validation Accuracy: {val_accuracy * 100:.2f}%'
            )

    '''
    Predict
    To return the most likely output class
    '''
    def predict(self, x):
        #Flatten the input to compare true values into 784 features
        x = self._flatten(x)
        with torch.no_grad():
            prob = self._forward_pass(x)
        predictions = torch.argmax(prob)
        return predictions

    def fit(self, train_loader, val_loader):
        start_time = time.time()

        for iteration in range(self.epochs):
            for x, y in train_loader:
                x = self._flatten(x)
                y = nn.functional.one_hot(y, 10)
                self.optimizer.zero_grad()


                output = self._forward_pass(x)
                self._backward_pass(y, output)
                self._update_weights()

            self._print_learning_progress(start_time, iteration, train_loader, val_loader)

    def compute_accuracy(self, data_loader):
        correct = 0
        for x, y in data_loader:
            pred = self.predict(x)
            correct += torch.sum(torch.eq(pred, y))

        return correct / len(data_loader.dataset)