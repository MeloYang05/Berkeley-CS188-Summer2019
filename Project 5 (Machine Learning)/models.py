import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(x, self.w)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        result = nn.as_scalar(self.run(x))
        if (result >= 0):
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 1
        correct = False
        while not correct:
            correct = True
            for x, y in dataset.iterate_once(batch_size):
                predict = self.get_prediction(x)
                if predict != nn.as_scalar(y):
                    correct = False
                    self.w.update(x, nn.as_scalar(y))


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.LearningRate = 0.05
        self.weight1 = nn.Parameter(1,300)
        self.bia1 = nn.Parameter(1,300)
        self.weight2 = nn.Parameter(300,100)
        self.bia2 = nn.Parameter(1,100)
        self.weight3 = nn.Parameter(100,50)
        self.bia3 = nn.Parameter(1,50)
        self.weight4 = nn.Parameter(50,1)
        self.bia4 = nn.Parameter(1,1)
        self.para_list = [self.weight1, self.weight2, self.weight3, self.weight4, 
                          self.bia1, self.bia2, self.bia3, self.bia4]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        first_coefficient = nn.Linear(x, self.weight1)
        first_predict = nn.AddBias(first_coefficient, self.bia1)
        first_layer = nn.ReLU(first_predict)
        second_coefficient = nn.Linear(first_layer, self.weight2)
        second_predict = nn.AddBias(second_coefficient, self.bia2)
        second_layer = nn.ReLU(second_predict)
        third_coefficient = nn.Linear(second_layer, self.weight3)
        third_predict = nn.AddBias(third_coefficient, self.bia3)
        third_layer = nn.ReLU(third_predict)
        output = nn.AddBias(nn.Linear(third_layer, self.weight4), self.bia4)
        return output

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 200
        loss = 0.01
        while loss >= 0.01:
            for x, y in dataset.iterate_once(batch_size):
                square_loss = self.get_loss(x, y)
                loss = nn.as_scalar(square_loss)
                gradients_list = nn.gradients(square_loss, self.para_list)
                for i in range(len(gradients_list)):
                    self.para_list[i].update(gradients_list[i], -self.LearningRate)
 

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.LearningRate = 0.05
        self.weight1 = nn.Parameter(784,300)
        self.bia1 = nn.Parameter(1,300)
        self.weight2 = nn.Parameter(300,100)
        self.bia2 = nn.Parameter(1,100)
        self.weight3 = nn.Parameter(100,50)
        self.bia3 = nn.Parameter(1,50)
        self.weight4 = nn.Parameter(50,10)
        self.bia4 = nn.Parameter(1,10)
        self.para_list = [self.weight1, self.weight2, self.weight3, self.weight4, 
                          self.bia1, self.bia2, self.bia3, self.bia4]


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        first_coefficient = nn.Linear(x, self.weight1)
        first_predict = nn.AddBias(first_coefficient, self.bia1)
        first_layer = nn.ReLU(first_predict)
        second_coefficient = nn.Linear(first_layer, self.weight2)
        second_predict = nn.AddBias(second_coefficient, self.bia2)
        second_layer = nn.ReLU(second_predict)
        third_coefficient = nn.Linear(second_layer, self.weight3)
        third_predict = nn.AddBias(third_coefficient, self.bia3)
        third_layer = nn.ReLU(third_predict)
        output = nn.AddBias(nn.Linear(third_layer, self.weight4), self.bia4)
        return output

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 200
        accuracy = 0
        while accuracy < 0.975:
            for x, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x, y)
                gradients_list = nn.gradients(loss, self.para_list)
                for i in range(len(gradients_list)):
                    self.para_list[i].update(gradients_list[i], -self.LearningRate)
            accuracy = dataset.get_validation_accuracy()

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.LearningRate = 0.05
        self.weight0 = nn.Parameter(self.num_chars, 300)
        self.bia0 = nn.Parameter(1,300)
        self.x_weight = nn.Parameter(self.num_chars, 300)
        self.h_weight = nn.Parameter(300,300)
        self.bia = nn.Parameter(1,300)
        self.output_weight = nn.Parameter(300, len(self.languages))
        self.output_bia = nn.Parameter(1, len(self.languages))
        self.para_list = [self.weight0, self.bia0, self.x_weight, self.h_weight,
                          self.bia, self.output_weight, self.output_bia]

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        for i in range(len(xs)):
            if i == 0:
                initial_coefficient = nn.Linear(xs[i], self.weight0)
                initial_predict = nn.AddBias(initial_coefficient, self.bia0)
                h = nn.ReLU(initial_predict)
            else:
                w_coefficient = nn.Linear(xs[i], self.x_weight)
                h_coefficient = nn.Linear(h, self.h_weight)
                combine_w_h = nn.Add(w_coefficient, h_coefficient)
                predict = nn.AddBias(combine_w_h, self.bia)
                h = nn.ReLU(predict)
        output_weight_coefficient = nn.Linear(h, self.output_weight)
        output = nn.AddBias(output_weight_coefficient, self.output_bia)
        return output


    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(xs), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 200
        accuracy = 0
        while accuracy < 0.9:
            for x, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x, y)
                gradients_list = nn.gradients(loss, self.para_list)
                for i in range(len(gradients_list)):
                    self.para_list[i].update(gradients_list[i], -self.LearningRate)
            accuracy = dataset.get_validation_accuracy()