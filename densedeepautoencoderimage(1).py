
import keras
from keras import layers

# This is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# This is our input image
input_img = keras.Input(shape=(784,))
x = layers.Dense(100, activation='relu')(input_img)
# "encoded" is the encoded representation of the input
encoded = layers.Dense(encoding_dim, activation='relu')(x)
x = layers.Dense(100, activation='relu')(encoded)
# "decoded" is the lossy reconstruction of the input
decoded = layers.Dense(784, activation='sigmoid')(x)

# This model maps an input to its reconstruction
autoencoder = keras.Model(input_img, decoded)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# This model maps an input to its encoded representation
encoder = keras.Model(input_img, encoded)

# This is our encoded input
encoded_input = keras.Input(shape=(encoding_dim,))
# Clone of the decoder hidden layer 
decoder_hidden = autoencoder.layers[-2](encoded_input)
# Retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1](decoder_hidden)


"""
we can do that also 
a = layers.Dense(100, activation='relu')(encoded_input)

decoded = layers.Dense(784, activation='sigmoid')(a)
"""
# Create the decoder model
decoder = keras.Model(encoded_input, decoder_layer)


from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()
print (x_train.shape)

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

autoencoder.fit(x_train, x_train,
                epochs=10, #50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))
"""
we didn’t train the encoder model

we didn’t train the decoder model
"""
# Encode and decode some digits
# Note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)



# Use Matplotlib (don't ask)
import matplotlib.pyplot as plt

n = 15  # How many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
#**********************************************************************************************************************************************************
#Copy (transfer) the weights

k = 0

for i in range (4,8):

    decoder.weights[k].assign(autoencoder.weights[i])

    k = k+1
#Modify the EncodedImage(s)

import random

imgsToBe = encoded_imgs[0:10]

for i in range(10):

    imgsToBe[i, 0:16] -= random.random()*5

    print (imgsToBe.shape)


#Reconstruct using the transferred Decoder
decoded_imgs = decoder.predict(imgsToBe)

n = 10

plt.figure(figsize=(20, 4))

for i in range(n):

    ax = plt.subplot(2, n, i + 1)

    plt.imshow(x_test[i].reshape(28, 28))

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i + 1 + n)

    plt.imshow(decoded_imgs[i].reshape(28, 28))

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.show()