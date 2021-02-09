import numpy as np


class Brain:
    """ Brain of the snake. The brain is based on a neural network. The input of the neural network is based on the
        vision of the snake. The output consists of 4 values. These 4 values are standing for the direction the snake
        can go in: ['left', 'right', 'up', 'down']. The highest value will determine which direction the snake will go
        in.

        Attributes:
            n_input (int): number of input values
            n_hidden_1 (int): number of neurons in first hidden layer
            n_hidden_2 (int): number of neurons in second hidden layer
            n_output (int): number of output values
            n_genes (int): number of genes in dna
            dna (np.array): Array of values. Each value represents a gene. The DNA contains the weight and bias values
                for the neural network
    """

    def __init__(self):
        # design neural network
        self.n_input = 24
        self.n_hidden_1 = 16
        self.n_hidden_2 = 16
        self.n_output = 4

        # determine number of weights and bias terms in neural network
        n_weights = self.n_input*self.n_hidden_1 + self.n_hidden_1*self.n_hidden_2 + self.n_hidden_2*self.n_output
        n_bias = self.n_hidden_1 + self.n_hidden_2 + self.n_output

        self.n_genes = n_weights + n_bias
        self.dna = self.initialize_dna()

    def initialize_dna(self):
        """ Get random values to initialize dna values

        Returns:
            np.array: Array with random values
        """
        return np.random.rand(1, self.n_genes) * 2 - 1

    def get_weights_from_dna(self):
        """ get weights and bias terms from DNA

        Returns:
            tuple: tuple containing 6 elements:
                1) weights for first hidden layer
                2) weights for second hidden layer
                3) weights for output layer
                4) bias terms for firs hidden layer
                5) bias terms for second hidden layer
                6) bias terms for output layer
        """

        W1_size = self.n_input*self.n_hidden_1
        W2_size = self.n_hidden_1*self.n_hidden_2
        W3_size = self.n_hidden_2*self.n_output

        start_W1, end_W1 = 0, W1_size
        start_B1, end_B1 = end_W1, end_W1 + self.n_hidden_1
        start_W2, end_W2 = end_B1, end_B1 + W2_size
        start_B2, end_B2 = end_W2, end_W2 + self.n_hidden_2
        start_W3, end_W3 = end_B2, end_B2 + W3_size
        start_B3, end_B3 = end_W3, end_W3 + self.n_output

        W1 = self.dna[:, start_W1:end_W1].reshape(self.n_hidden_1, self.n_input)
        B1 = self.dna[:, start_B1:end_B1].reshape(self.n_hidden_1, 1)
        W2 = self.dna[:, start_W2:end_W2].reshape(self.n_hidden_2, self.n_hidden_1)
        B2 = self.dna[:, start_B2:end_B2].reshape(self.n_hidden_2, 1)
        W3 = self.dna[:, start_W3:end_W3].reshape(self.n_output, self.n_hidden_2)
        B3 = self.dna[:, start_B3:end_B3].reshape(self.n_output, 1)

        return W1, W2, W3, B1, B2, B3

    def forward_propagation(self, X):
        """ Forward propagation of network

        Args:
            X (np.array): Array containing the input values for the neural network

        Returns:
            np.array: Array of output values of the neural network
        """
        W1, W2, W3, B1, B2, B3 = self.get_weights_from_dna()

        Z1 = np.dot(W1, X) + B1
        A1 = np.maximum(Z1, 0)
        Z2 = np.dot(W2, A1) + B2
        A2 = np.maximum(Z2, 0)
        Z3 = np.dot(W3, A2) + B3
        A3 = Z3
        return A3


