Machine Learning - Homework 3
William Pride - wsp26
Brandon Cook - bmc85

Our implementation of the primal perceptron algorithm uses two classes, Example and perceptron.

Example stores a representation of an instance or weight vector and contains the necessary functions

Storage is done using a Hashtable for linear lookup time.

Example has two constructors
	One creates an empty Hashtable which is populated by the perceptron class, used for instances
	One creates a Hashtable with entries of (i,Integer(0)) for all possible features, used for w

perceptron contains all static methods used for the creation and manipulation of Examples

The main working function is 

iterativePerceptron(String tfile, String vFile, double mu, int its) where
	tFile is the name of a file to generate the training set from
	vFile is the name of a file to validate the weight vector with
	mu is the desired learning rate
	its is the number of training iterations to perform

		Returns the Example w trained from tfile
		prints out the error rate on the training and validation sets after each iteration

iterativePerceptron uses
	improveW() to improve the weight vector
	paValidation() to test the weight vector

improveW(Example w, Example x, double mu, double y, double dp)
	w is the weight vector to be updated
	x is the misclassified instance used to modify w
	mu is the learning rate
	dp is the value for x predicted by w
	y is the actual classification of x

	Proceeds using update rules form the primal perceptron algorithm and returns the new w vector

paValidation(Example w, ArrayList<Example> validate, double mu)
	w is the weight vector to test
	validate is the validation set
	mu is the learning rate

	Iterates through every instance in validate, 
	calculates the value guessed by w, 
	determines whether w guessed correctly,
	Prints the error rate
	Returns the weight vector

USE

	An example call to the interactions pane might be

	perceptron.iterativePerceptron("polarity.train","polarity.validation",1,20)

	Where we want the perceptron trained on the instances in "polarity.train" with a learning rate of 1 and 20 learning iterations, tested on the instances in "polarity.validation".
		Prints the training and validation error rates after each iteration
		Returns the weight vector w
