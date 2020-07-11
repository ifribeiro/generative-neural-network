"""
Code based on <https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-1-dimensional-function-from-scratch-in-keras/>
"""

from numpy.random import rand
from numpy import hstack
from numpy import ones,zeros

#imports for the discriminator model
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model

import matplotlib.pyplot as plt

 
# generate n real samples with class labels
def generate_real_samples(n=100):
	# generate random inputs in range [-0.5, 0.5]
	X1 = rand(n) - 0.5
	# generate outputs X^2 (quadratic)
	X2 = X1 * X1
	# stack arrays
	X1 = X1.reshape(n, 1)
	X2 = X2.reshape(n, 1)
	X = hstack((X1, X2))

	#generate class labels
	y = ones((n,1))

	return X,y

# TODO: A função que gera os fakes tem que gerar valores possíveis de serem gerados pela real

#generate n fake samples with class labels
def generate_fake_samples(n):
	# generate inputs in [-1, 1]
	X1 = -1 + rand(n) * 2
	# generate outputs in [-1, 1]
	X2 = -1 + rand(n) * 2
	# stack arrays
	X1 = X1.reshape(n,1)
	X2 = X2.reshape(n,1)
	X = hstack((X1,X2))
	# generate class labels
	y = zeros((n, 1))

	return X,y

# generate n randoms samples from x^2
def generate_samples(n=100):
	# generate random inputs in [-0.5, 0.5]
	X1 = rand(n) - 0.5
	# generate outputs X^2 (quadratic)
	X2 = X1 * X1
	# stack arrays
	X1 = X1.reshape(n, 1)
	X2 = X2.reshape(n, 1)

	return hstack((X1, X2))

# function to train the discriminator
def train_discriminator(model, n_epochs=100, n_batch=128):
	"""
	Train the discriminator

	Parameters
	----------
	model: Model (from keras.models)
		Model that will be used to train the discriminator
	n_epochs: int
		Number of epochs that the discriminator will be trained
	n_batch = int
		Number of batchs
	"""
	real = []
	fake = []

	half_batch = int(n_batch/2)
	#run epochs manually
	for i in range(n_epochs):
		# generate real examples
		X_real, y_real = generate_real_samples(half_batch)
		# update model
		model.train_on_batch(X_real, y_real)

		# generate fake examples
		X_fake, y_fake = generate_fake_samples(half_batch)
		# update model
		model.train_on_batch(X_fake,y_fake)

		# evaluate the model
		_, acc_real = model.evaluate(X_real, y_real,verbose=0)
		_, acc_fake = model.evaluate(X_fake, y_fake,verbose=0)
		
		real.append(acc_real)
		fake.append(acc_fake)

	return real, fake




# define the standalone discriminator model
def define_discriminator(n_inputs=2):
	model = Sequential()

	"""
	TODO: ler ReLU activation function
	- One hidden layer with 25 nodes using ReLU activation function
	TODO: ler He weight ininitalization method
	- the He weight initialization method
	"""
	
	model.add(Dense(25,activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))

	"""
	- One hidden layer with 1 node using Sigmoid activation function
	"""
	model.add(Dense(1,activation='sigmoid'))

	#compile model
	model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
	
	return model



# define the discriminator model
model = define_discriminator()

print ("Fitting the model")

# fit the model
results_real, results_fake = train_discriminator(model, n_epochs=1000)

# plotting the GAN's accuracy to discriminates between fake and real data
fig,ax = plt.subplots(figsize=(12,5), ncols=2,nrows=1)
xs = range(len(results_real))

ax[0].plot(xs,results_real, color='green') # real inputs
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Accuracy')

ax[1].plot(xs,results_fake, color='red') # fake inputs
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy')

plt.show()