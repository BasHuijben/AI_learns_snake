import numpy as np

from functions.snake_board import SnakeBoard
from functions.brain import Brain


class Snake(SnakeBoard, Brain):
    """ Generate new snake. This class inherits from two parent classes:
        1) SnakeBoard: Generates a board where the snake can play on
        2) Brain: Generates the brain of the snake

    Attributes:
        snake_head (np.array): 1 by 2 array containing the coordinates of the snakes head
        snake_body (np.array): x by 2 array containing the coordinates of the body parts. X = the number of body parts
        fitness (float): Fitness of the snake
        alive (bool): If the snake is still alive or not
        moves_played (int): Number of moves played before the snake dies
        moves_without_apple (int): Number of moves played before an apple is found.
            Every time an apple is found this number will reset to 0
        decisions (list): The decisions a snake can make; ['left', 'right', 'up', 'down']
        direction (str): Current direction of the snake; 'left', 'right', 'up', 'down'
        total_snake_length (int): Total length of the snake
        apple_found (bool): True if the snake has found an apple
        total_apples_found (int): Total number of apples found
    """

    def __init__(self):
        SnakeBoard.__init__(self)
        Brain.__init__(self)

        self.snake_head = np.asarray([[5, 5]])
        self.snake_body = np.asarray([[5, 4], [5, 3]])
        self.fitness = 0
        self.alive = True
        self.moves_played = 0
        self.moves_without_apple = 0
        self.decisions = ['left', 'right', 'up', 'down']
        self.direction = 'right'
        self.total_snake_length = 3
        self.apple_found = False
        self.total_apples_found = 0

    def reset_snake(self):
        """ Reset snakes initial values"""
        self.snake_head = np.asarray([[5, 5]])
        self.snake_body = np.asarray([[5, 4], [5, 3]])
        self.fitness = 0
        self.alive = True
        self.moves_played = 0
        self.moves_without_apple = 0
        self.direction = 'right'
        self.total_snake_length = 3
        self.apple_found = False
        self.total_apples_found = 0
        self.n_apples = 0
        self.get_new_apple()

    def snake_move(self):
        self.make_decision()

        self.update_snake()
        self.found_apple()
        self.snake_alive()

    def make_decision(self):
        """ Change the direction of the snake
        """
        A3 = self.forward_propagation(self.get_vision())
        self.direction = self.decisions[np.argmax(A3)]

    def snake_alive(self):
        """ Determine if the snake is alive or not. The snake can die in 3 different ways:
            1) When the snake hits the edges of the board
            2) When the snake hits his own body
            3) When number of moves without an apple is larger than the number of grid cells
        """
        if np.any(self.snake_head == -1) or np.any(self.snake_head == self.grid_size):
            self.alive = False
        if np.any(np.all(self.snake_head == self.snake_body, axis=1)):
            self.alive = False
        if self.moves_without_apple > self.grid_size**2:
            self.alive = False

        if not self.alive:
            self.determine_fitness()

    def determine_fitness(self):
        """When the snake dies its fitness score is calculated"""
        # Fitness score copied from https://chrispresso.io/AI_Learns_To_Play_Snake
        # 1. Reward snakes early on for exploration + finding a couple apples.
        # 2. Have an increasing reward for snakes as they find more apples.
        # 3. Penalize snakes for taking a lot of steps.Putting those rules into code looks something like this
        self.fitness = self.moves_played \
                       + ((2**self.total_apples_found) + 500 * (self.total_apples_found**2.1)) \
                       - ((0.25 * self.moves_played)**1.3 * (self.total_apples_found**1.2))

    def found_apple(self):
        """ Determine if snake has found an apple. If found set apple_found to True, increase number of apple founds,
            get a new apple and set moves_without_apple to 0
        """
        if np.all(self.snake_head == self.apple):
            self.apple_found = True
            self.total_apples_found += 1
            self.get_new_apple()
            self.moves_without_apple = 0

    def add_body_part(self):
        """ Add body part to snake
        """
        if self.snake_body is None:
            self.snake_body = self.snake_head
        else:
            self.snake_body = np.append(self.snake_head, self.snake_body, axis=0)
        self.total_snake_length += 1

    def remove_tail(self):
        """ Remove last body part.
        """
        if self.snake_body is not None:
            self.snake_body = self.snake_body[:-1]

    def update_snake(self):
        """ Update position of snake
        """
        if self.apple_found:
            self.add_body_part()
            self.apple_found = False
        else:
            self.update_body()
        self.update_head()

        self.moves_played += 1
        self.moves_without_apple += 1

    def update_head(self):
        """ Update head. Helper function for update_snake to update the position of the head
        """
        if self.direction == 'right':
            self.snake_head[0, 1] += 1
        elif self.direction == 'left':
            self.snake_head[0, 1] -= 1
        elif self.direction == 'up':
            self.snake_head[0, 0] -= 1
        elif self.direction == 'down':
            self.snake_head[0, 0] += 1
        else:
            raise("Direction %s is unknown" % self.direction)

    def update_body(self):
        """ Update body. Helper function for update_snake to update the position of the body
        """
        self.add_body_part()
        self.remove_tail()

    def get_vision(self):
        """ Get vision of snake. The snake can see 3 things in 4/8 different directions and has a sense of the
            direction it is going. It can see the following three things:
                1) In 8 direction it can see if an apple is present (binary)
                2) In 4 direction it can see the distance to wall
                3) In 8 direction it can see if a body part is near (binary)
        """
        vision = np.concatenate((self.get_apple_vision(),
                                 self.get_edge_vision(),
                                 self.get_body_vision(),
                                 self.get_direction_vision()), axis=None)
        return vision.reshape(-1, 1)

    def get_direction_vision(self):
        """ One hot encode the direction of the snake

        Returns:
            np.array: contains 4 binary elements. If 1, than the snake is going in that direction
        """
        return np.asarray([self.direction == decision for decision in self.decisions])

    def get_body_vision(self):
        """ Determine if a body part is near to its head in 8 directions (binary).

        Returns:
            np.array: contains 8 binary elements. If 1, than a body part is near in that direction
        """
        body_diff = self.snake_body - self.snake_head
        coordinates_around_head = np.asarray([(i_row, i_col) for i_row in [-1, 0, 1]
                                                             for i_col in [-1, 0, 1]
                                                             if not (i_row == 0 and i_col == 0)])
        body_in_sight = np.asarray([np.any(np.all(body_diff == coordinates, axis=1))
                                    for coordinates in coordinates_around_head])

        return body_in_sight

    def get_edge_vision(self):
        """ Get distance to edge in four directions

        Returns:
            np.array: contains 8 values between 0 and 1. If 1 the edge is near, if 0 the edge is as far away as possible
        """
        size = (self.grid_size - 1)
        edge_left = 1 - (self.snake_head[0, 1] / size)
        edge_right = 1 - ((size - self.snake_head[0, 1]) / size)
        edge_up = 1 - (self.snake_head[0, 0] / size)
        edge_down = 1 - ((size - self.snake_head[0, 0]) / size)
        return np.asarray([edge_left, edge_right, edge_up, edge_down])

    def get_apple_vision(self):
        """Determine if the snake can see the apple in 8 directions

        Returns:
            np.array: contains 8 binary values. If 1 the snake can see apple in that direction. Otherwise 0.
        """
        apple_diff = self.snake_head - self.apple
        apple_in_sight = np.zeros((1, 8))
        if apple_diff[0, 0] == 0 and apple_diff[0, 1] > 0:
            apple_in_sight[0, 0] = 1
        elif apple_diff[0, 0] == 0 and apple_diff[0, 1] < 0:
            apple_in_sight[0, 1] = 1
        elif apple_diff[0, 1] == 0 and apple_diff[0, 1] > 0:
            apple_in_sight[0, 2] = 1
        elif apple_diff[0, 1] == 0 and apple_diff[0, 1] < 0:
            apple_in_sight[0, 3] = 1
        elif abs(apple_diff[0, 0]) == abs(apple_diff[0, 1]) and apple_diff[0, 0] < 0:
            apple_in_sight[0, 4] = 1
        elif abs(apple_diff[0, 0]) == abs(apple_diff[0, 1]) and apple_diff[0, 0] > 0:
            apple_in_sight[0, 5] = 1
        elif abs(apple_diff[0, 0]) == abs(apple_diff[0, 1]) and apple_diff[0, 1] < 0:
            apple_in_sight[0, 6] = 1
        elif abs(apple_diff[0, 0]) == abs(apple_diff[0, 1]) and apple_diff[0, 1] > 0:
            apple_in_sight[0, 7] = 1

        return apple_in_sight
