import time
import numpy as np
import scratch.utils as utils
from scratch.lr_scheduler import cosine_annealing


class Network():
    def __init__(self, sizes, epochs=50, learning_rate=0.01, random_state=1):
        self.sizes = sizes
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.activation_func = utils.sigmoid
        self.activation_func_deriv = utils.sigmoid_deriv
        self.output_func = utils.softmax
        self.output_func_deriv = utils.softmax_deriv
        self.cost_func = utils.mse
        self.cost_func_deriv = utils.mse_deriv

        self.params = self._initialize_weights()

        #ADD: need to cache hidden layer inputs and outputs for backprop
        self.a3 = np.zeros(self.sizes[3])
        self.h2 = np.zeros(self.sizes[2])
        self.a2 = np.zeros(self.sizes[2])
        self.h1 = np.zeros(self.sizes[1])
        self.a1 = np.zeros(self.sizes[1])
        self.x = np.zeros(self.sizes[0])


    def _initialize_weights(self):
        # number of neurons in each layer
        input_layer = self.sizes[0]
        hidden_layer_1 = self.sizes[1]
        hidden_layer_2 = self.sizes[2]
        output_layer = self.sizes[3]

        # random initialization of weights
        np.random.seed(self.random_state)
        params = {
            'W1': np.random.rand(hidden_layer_1, input_layer) - 0.5,
            'W2': np.random.rand(hidden_layer_2, hidden_layer_1) - 0.5,
            'W3': np.random.rand(output_layer, hidden_layer_2) - 0.5,
        }

        return params


    def _forward_pass(self, x_train):
        '''
        DONE: Implement the forward propagation algorithm.
        The method should return the output of the network.
        '''
        # Save x_train for backprop
        self.x = x_train
        
        # 1. Input to hidden 1
        self.a1 = self.params['W1'] @ x_train #weighted sum = W1@X (no bias!)
        self.h1 = self.activation_func(self.a1) #sigmoid activation: h1i ∈ (0,1)

        # 2. Hidden 1 to hidden 2
        self.a2 = self.params['W2'] @ self.h1 #weighted sum
        self.h2 = self.activation_func(self.a2) #sigmoid

        # 3. Hidden 2 to output activation
        self.a3 = self.params['W3'] @ self.h2 #weighted sum
        output = self.output_func(self.a3) #softmax output: sum(output) = 1.0

        return output


    def _backward_pass(self, y_train, output):
        '''
        DONE: Implement the backpropagation algorithm responsible for updating the weights of the neural network.
        The method should return a dictionary of the weight gradients which are used to update the weights in self._update_weights().
        '''  
        # Count backwards! all derivative variables are dL/...
        
        # 3. Loss to output to hidden 2 weights
        dy = self.cost_func_deriv(y_train, output) # (sizes[3],) #mse loss
        da3 = dy * self.output_func_deriv(self.a3) # (sizes[3],) #softmax output
        dw3 = da3[:, np.newaxis] @ self.h2[np.newaxis, :] # (sizes[3], sizes[2]) #weighted sum
        
        # 2. Hidden 2 to hidden 1 weights
        dh2 = da3 @ self.params['W3'] # (sizes[2],) #weighted sum
        da2 = dh2 * self.activation_func_deriv(self.a2) # (sizes[2],) #sigmoid
        dw2 = da2[:, np.newaxis] @ self.h1[np.newaxis, :] # (sizes[2], sizes[1]) #weighted sum

        # 1. Hidden 1 to input weights
        dh1 = da2 @ self.params['W2'] # (sizes[1],) #weighted sum
        da1 = dh1 * self.activation_func_deriv(self.a1) # (sizes[1],) #sigmoid
        dw1 = da1[:, np.newaxis] @ self.x[np.newaxis, :] # (sizes[1], sizes[0]) #weighted sum

        # Create gradient: all of the partial derivatives to update weights
        weights_gradient = {
            'dW1': dw1,
            'dW2': dw2,
            'dW3': dw3,
        }
        
        return weights_gradient


    def _update_weights(self, weights_gradient, learning_rate):
        '''
        DONE: Update the network weights according to stochastic gradient descent.
        '''
        # Update each set of weights by its negative gradient * learning_rate
        self.params['W1'] -= learning_rate * weights_gradient['dW1']
        self.params['W2'] -= learning_rate * weights_gradient['dW2']
        self.params['W3'] -= learning_rate * weights_gradient['dW3']
        
        return None


    def _print_learning_progress(self, start_time, iteration, x_train, y_train, x_val, y_val):
        train_accuracy = self.compute_accuracy(x_train, y_train)
        val_accuracy = self.compute_accuracy(x_val, y_val)
        print(
            f'Epoch: {iteration + 1}, ' \
            f'Training Time: {time.time() - start_time:.2f}s, ' \
            f'Training Accuracy: {train_accuracy * 100:.2f}%, ' \
            f'Validation Accuracy: {val_accuracy * 100:.2f}%'
            )


    def compute_accuracy(self, x_val, y_val):
        predictions = []
        for x, y in zip(x_val, y_val):
            pred = self.predict(x)
            predictions.append(pred == np.argmax(y))

        return np.mean(predictions)


    def predict(self, x):
        '''
        DONE: Implement the prediction making of the network.
        The method should return the index of the most likeliest output class.
        '''
        #run forward pass to get the softmax output
        output = self._forward_pass(x)

        #output "prediction" (returns only first index of largest value in case of ties!)
        return np.argmax(output)

    def fit(self, x_train, y_train, x_val, y_val, cosine_annealing_lr=False):

        start_time = time.time()

        for iteration in range(self.epochs):
            for x, y in zip(x_train, y_train):
                
                if cosine_annealing_lr:
                    learning_rate = cosine_annealing(self.learning_rate, 
                                                     iteration, 
                                                     len(x_train), 
                                                     self.learning_rate)
                else: 
                    learning_rate = self.learning_rate
                output = self._forward_pass(x)
                weights_gradient = self._backward_pass(y, output)
                
                self._update_weights(weights_gradient, learning_rate=learning_rate)

            self._print_learning_progress(start_time, iteration, x_train, y_train, x_val, y_val)
