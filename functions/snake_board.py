import numpy as np


class SnakeBoard:
    """ Snake board

    Attributes:
        grid_size (int): Size of the grid
        n_apples (int): Number of apples found
        apples (list): List of apples that will appear in the game
        apple (np.array): Location of the current apple. x and y coordinates of the grid.
    """

    def __init__(self):
        self.grid_size = 10
        self.n_apples = 0
        self.apples = self.generate_apples()
        self.apple = 0
        self.get_new_apple()

    def generate_apples(self):
        """ Generate list with apples

        Returns:
            list: List with the location of the apples that will appear in the game
        """
        np.random.seed(2)
        apples = np.random.randint(0, self.grid_size, size=(self.grid_size**2, 2))
        np.random.seed(None)
        return apples

    def get_new_apple(self):
        """ Get new apple from list of apples
        """
        self.apple = self.apples[self.n_apples].reshape((1, -1))
        self.n_apples += 1
