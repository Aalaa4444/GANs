import keras
from keras import layers
###########################
# This is the size of our encoded representations
encoding_dim = 32  

# This is our input image
input_img = keras.Input(shape=(784,))
# "encoded" is the encoded representation of the input
encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = layers.Dense(784, activation='sigmoid')(encoded)

# This model maps an input to its reconstruction
autoencoder = keras.Model(input_img, decoded)
###############################
# This model maps an input to its encoded representation
encoder = keras.Model(input_img, encoded)
################################
# This is our encoded (32-dimensional) input
encoded_input = keras.Input(shape=(encoding_dim,))
# Retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# Create the decoder model
decoder = keras.Model(encoded_input, decoder_layer(encoded_input))
##################################
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
##################################
from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()
print (x_train.shape)
#################################
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)
#################################
autoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))
#################################
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)
####################################
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
###################################
import matplotlib.pyplot as plt

n = 15  # How many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display original
    ax = plt.subplot(3, n, i + 1+n)
    plt.imshow(encoded_imgs[i].reshape(8, 4))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(3, n, i + 1 + 2*n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
#####################################
b = [0,3]
imgs = (x_test[b])
encoded_imgs = encoder.predict(imgs)
copyofLatent = encoded_imgs
decoded_imgs = decoder.predict(copyofLatent)
######################################
import matplotlib.pyplot as plt

n = 2  # How many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(encoded_imgs[i].reshape(8, 4))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(3, n, i + 1 + 2*n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
#########################################
interpolation_steps = 20
interpolated_encodings = np.linspace(copyofLatent[0], copyofLatent[1], interpolation_steps)
print(interpolated_encodings.shape)
print(interpolated_encodings[1])
###########################################
n = 20  # How many digits we will display
plt.figure(figsize=(10, 2))
for i in range(n):
    # Display reconstruction
    ax = plt.subplot(1, n, i+1)
    plt.imshow(interpolated_encodings[i].reshape(8, 4))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
##########################################
decoded_imgs = decoder.predict(interpolated_encodings)
##########################################
n = 20  # How many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display reconstruction
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(interpolated_encodings[i].reshape(8, 4))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
##########################################
b = [3,9, 14 , 23]
imgs2 = (x_test[b])
encoded_imgs2 = encoder.predict(imgs2)
copyofLatent2 = encoded_imgs2
decoded_imgs2 = decoder.predict(copyofLatent2)
############################################
n = 4  # How many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(imgs2[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(encoded_imgs2[i].reshape(8, 4))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(3, n, i + 1 + 2*n)
    plt.imshow(decoded_imgs2[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
###############################################
interpolation_steps = 4
interpolated_encodings2 = np.linspace(
    np.linspace(copyofLatent2[0], copyofLatent2[1], interpolation_steps),
    np.linspace(copyofLatent2[2], copyofLatent2[3], interpolation_steps),
    interpolation_steps,
)
print (interpolated_encodings2.shape)
#################################################
ii = interpolated_encodings2.reshape(16,32)
#################################################
n = 16  # How many digits we will display
plt.figure(figsize=(25, 5))
a=1
for i in range(n):
    # Display reconstruction
    ax = plt.subplot(4, 4, a)
    plt.imshow(ii[i].reshape(8, 4))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    a = a+1
plt.show()
###############################################
decoded_imgs2 = decoder.predict(ii)
###############################################
n = 16  # How many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display reconstruction
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(ii[i].reshape(8, 4))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs2[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
################################################
