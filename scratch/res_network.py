import time
import numpy as np
from scratch.network import Network

class ResNetwork(Network):

    '''
    We have 1 input layer and 2 hidden layers (h1, h2) from Network. 
    For the residual skip, we take the output of the first h1 (linear layer and activation) plus the input itself, and added to h2 so that it can learn the residual.
    '''

    def __init__(self, sizes, epochs=50, learning_rate=0.01, random_state=1):
        # Inherit the classes, but adding a logic for the residual
        # Our Network(sizes=[784, 128, 64, 10], if checks [input, h1, h2, output] so that it can be valid. We included no biases.
        super(ResNetwork, self).__init__(
            sizes=sizes,
            epochs=epochs,
            learning_rate=learning_rate,
            random_state=random_state
            )
        
        if len(sizes) < 4:
            raise ValueError("At least 2 Hidden Layers are needed for set up.")

    def _forward_pass(self, x_train):
        # First h1
        self.z1 = self.params['W1'] @ x_train 
        self.h1 = self.activation_func(self.z1)

        # h2 with Residual
        self.z2 = self.params['W2'] @ self.h1 
        self.h2 = self.activation_func(self.z2) + self.h1 #Skips the connection

        # Output Layer
        self.z3 = self.params['W3'] @ self.h2 
        self.output = self.output_func(self.z3)

        return self.output 

    def _backward_pass(self, y_train, output):
        '''
        Backpropagation implementation for the residual network. 
        We created a dictionary of gradients for W1, W2, W3.
        '''
        # All derivative variables are dL/...
        
        # Output Layer and first h1
        dy = self.cost_func_deriv(y_train, output) # (sizes[3],) #mse loss
        dz3 = dy * self.output_func_deriv(self.z3) # (sizes[3],) #softmax output
        dw3 = dz3[:, np.newaxis] @ self.h2[np.newaxis, :] # (sizes[3], sizes[2]) #weighted sum
        dh2 = dz3 @ self.params['W3'] # (sizes[2],) #weighted sum

        # h2 with Residual
        dz2 = (dh2 * self.activation_func_deriv(self.z2)) # (sizes[2],) #sigmoid
        dw2 = dz2[:, np.newaxis] @ self.h1[np.newaxis, :] # (sizes[2], sizes[1]) #weighted sum
        backpropagation = (dz2 @self.params['W2']) + dh2 # total gradient through W2, with skip connection

        # h1 to input weights
        dz1 = backpropagation * self.activation_func_deriv(self.z1) #pushes backpropagation through sigmoid
        dw1 = dz1[:, np.newaxis] @ self.x[np.newaxis, :] 

        # Create gradient: all of the partial derivatives to update weights
        weights_gradient = {
            'dW1': dw1,
            'dW2': dw2,
            'dW3': dw3,
        }
        
        return weights_gradient