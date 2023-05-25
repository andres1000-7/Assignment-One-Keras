import keras
from keras.models import Sequential
from keras.layers import Activation,Dense,Dropout,Conv2D,Flatten,MaxPooling2D
from keras.datasets import cifar100
from keras import optimizers
from matplotlib import pyplot as plt
import numpy as np
from keras import backend as K
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.callbacks import Callback, EarlyStopping
import tensorflow as tf
import datetime, os
from keras.utils import to_categorical
import time

# Set the desired dataset sizes and increment value
dataset_sizes = [100, 200, 300, 400, 500]
num_layers_list = [3, 4, 5, 6]  # Number of layers in each model
increment = 15
num_iterations = 10

batch_size = 128
num_classes = 100
epochs = 20

# Load the CIFAR-100 dataset
(X_train, y_train), (X_test, y_test) = cifar100.load_data()


# Function to increase the dataset for a given dataset size
def increase_dataset(dataset_size):
    augmented_images = []
    augmented_labels = []

    # Iterate over each class
    for class_label in range(100):
        # get the images and labels from the class
        class_images = X_train[y_train.flatten() == class_label]
        class_labels = y_train[y_train.flatten() == class_label]

        # Select the desired number of images for the current class
        selected_images = class_images[:dataset_size]

        # Increment the dataset size by adding randomly selected images from the current class
        while len(selected_images) < dataset_size:
            # randomly select additional images from the current class
            additional_images = np.random.choice(class_images, increment)
            # join selected_images and additional_images
            selected_images = np.concatenate((selected_images, additional_images), axis=0)

        # append the selected images and class labels to the augmented empty lists
        augmented_images.append(selected_images)
        augmented_labels.append(class_labels[:dataset_size])

    # return: concatenate the arrays within the list
    return np.concatenate(augmented_images), np.concatenate(augmented_labels)


# Preprocess the dataset
def preprocess_dataset(X_train, y_train, X_test, y_test):
    # Convert pixel values to float and normalize
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Convert labels to categorical format
    y_train = to_categorical(y_train, 100)
    y_test = to_categorical(y_test, 100)

    return X_train, y_train, X_test, y_test


def create_model(num_layers):
    model = Sequential()

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=(32,32,3)))
    model.add(MaxPooling2D((2, 2)))

    for _ in range(num_layers - 2):
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

    # model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model

# Load the TensorBoard notebook extension
%reload_ext tensorboard

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=8, restore_best_weights = True)

# For the TensorBoard, set the directory you want to save your log.
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) # log directory will be created
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

# # Iterate over each dataset size
# for size in dataset_sizes:
#     print(f"Dataset size: {size}")

#     # Augment the dataset
#     X_augmented, y_augmented = increase_dataset(size)

#     # Preprocess the augmented dataset
#     X_train_processed, y_train_processed, X_test_processed, y_test_processed = preprocess_dataset(
#         X_augmented, y_augmented, X_test, y_test)

#     print('x_train shape:', X_train_processed.shape)
#     print(X_train_processed.shape[0], 'train samples')
#     print(X_test_processed.shape[0], 'test samples')
#     print()

for num_layers in num_layers_list:
    print(f"Number of layers: {num_layers}")

    X_augmented, y_augmented = increase_dataset(500)
    X_train_processed, y_train_processed, X_test_processed, y_test_processed = preprocess_dataset(X_augmented, y_augmented,
                                                                                                  X_test, y_test)

    # Create the model
    model = create_model(num_layers)

    # Compile the model
    model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate = 0.0001), metrics=['accuracy'])
    model.summary()

# largest_dataset_size = max(dataset_sizes)
# error_metrics = []
# time_taken = []

# for num_layers in num_layers_list:
#     print(f"Number of layers: {num_layers}")

#     X_augmented, y_augmented = increase_dataset(largest_dataset_size)
#     X_train_processed, y_train_processed, X_test_processed, y_test_processed = preprocess_dataset(X_augmented, y_augmented,
#                                                                                                   X_test, y_test)

#     # Create the model
#     model = create_model(num_layers)

#     # Compile the model
#     model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate = 0.0001), metrics=['accuracy'])

#     # Train the model
#     start_time = time.time()
#     history = model.fit(X_train_processed, y_train_processed, batch_size=64, epochs=20,
#                         validation_data=(X_test_processed, y_test_processed))
#     end_time = time.time()

#     # Calculate the model size in millions of parameters
#     model_size = model.count_params() / 1_000_000

#     score = model.evaluate(X_test_processed, y_test_processed, verbose=0)
#     print('Test loss:', score[0])
#     print('Test accuracy:', score[1])

#     # Plot the error metrics (loss and accuracy)
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(history.history['loss'], label='Training Loss')
#     plt.plot(history.history['val_loss'], label='Validation Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.title('Error Metrics vs. Model Size')
#     plt.legend()

#     plt.subplot(1, 2, 2)
#     plt.plot(history.history['accuracy'], label='Training Accuracy')
#     plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.title('Error Metrics vs. Model Size')
#     plt.legend()

#     plt.tight_layout()
#     plt.show()

#     # Store the error metrics for the largest dataset size
#     error_metrics.append((model_size, history.history['loss'][-1], history.history['accuracy'][-1]))

#     # Store the time taken for training
#     time_taken.append((model_size, end_time - start_time))

# # Plot the error metrics vs. model size
# plt.figure(figsize=(8, 6))
# model_sizes = [x[0] for x in error_metrics]
# loss_values = [x[1] for x in error_metrics]
# accuracy_values = [x[2] for x in error_metrics]

# plt.plot(model_sizes, loss_values, label='Loss')
# plt.plot(model_sizes, accuracy_values, label='Accuracy')
# plt.xlabel('Model Size (Millions of Parameters)')
# plt.ylabel('Error Metrics')
# plt.title('Error Metrics vs. Model Size (Largest Dataset Size)')
# plt.legend()
# plt.show()

# # Plot the time taken vs. model size
# plt.figure(figsize=(8, 6))
# model_sizes = [x[0] for x in time_taken]
# time_values = [x[1] for x in time_taken]

# plt.plot(model_sizes, time_values)
# plt.xlabel('Model Size (Millions of Parameters)')
# plt.ylabel('Time Taken (Seconds)')
# plt.title('Time Taken vs. Model Size (Largest Dataset Size)')
# plt.show()

# largest_network_size = 6
# error_metrics = []
# time_taken = []

# for dataset_size in dataset_sizes:
#     print(f"Dataset size: {dataset_size}")

#     X_augmented, y_augmented = increase_dataset(dataset_size)
#     X_train_processed, y_train_processed, X_test_processed, y_test_processed = preprocess_dataset(X_augmented, y_augmented,
#                                                                                                   X_test, y_test)

#     # Create the model
#     model = create_model(largest_network_size)

#     # Compile the model
#     model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate = 0.0001), metrics=['accuracy'])

#     # Train the model
#     start_time = time.time()
#     history = model.fit(X_train_processed, y_train_processed, batch_size=128, epochs=20,
#                         validation_data=(X_test_processed, y_test_processed))
#     end_time = time.time()

#     # Plot the error metrics (loss and accuracy)
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(history.history['loss'], label='Training Loss')
#     plt.plot(history.history['val_loss'], label='Validation Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.title('Error Metrics vs. Model Size')
#     plt.legend()

#     plt.subplot(1, 2, 2)
#     plt.plot(history.history['accuracy'], label='Training Accuracy')
#     plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.title('Error Metrics vs. Model Size')
#     plt.legend()

#     plt.tight_layout()
#     plt.show()

#     dataset_size *= 100
#     # Store the error metrics for the largest dataset size
#     error_metrics.append((dataset_size, history.history['loss'][-1], history.history['accuracy'][-1]))

#     # Store the time taken for training
#     time_taken.append((dataset_size, end_time - start_time))

# # Plot the error metrics vs. model size
# plt.figure(figsize=(8, 6))
# data_sizes = [x[0] for x in error_metrics]
# loss_values = [x[1] for x in error_metrics]
# accuracy_values = [x[2] for x in error_metrics]

# plt.plot(data_sizes, loss_values, label='Loss')
# plt.plot(data_sizes, accuracy_values, label='Accuracy')
# plt.xlabel('Dataset size')
# plt.ylabel('Error Metrics')
# plt.title('Error Metrics vs. Dataset size (Largest Network Size)')
# plt.legend()
# plt.show()

# # Plot the time taken vs. model size
# plt.figure(figsize=(8, 6))
# data_sizes = [x[0] for x in time_taken]
# time_values = [x[1] for x in time_taken]

# plt.plot(data_sizes, time_values)
# plt.xlabel('Dataset size')
# plt.ylabel('Time Taken (Seconds)')
# plt.title('Time Taken vs. Dataset size (Largest Network Size)')
# plt.show()

# largest_dataset_size = max(dataset_sizes)
# largest_network_size = 6

# # Lists to store results
# error_metrics = []
# iterations = []
# print("Largest: ", largest_dataset_size, largest_network_size)

# X_augmented, y_augmented = increase_dataset(largest_dataset_size)
# X_train_processed, y_train_processed, X_test_processed, y_test_processed = preprocess_dataset(X_augmented, y_augmented,
#                                                                                               X_test, y_test)

# # Create the model
# model = create_model(largest_network_size)
# model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate = 0.0001), metrics=['accuracy'])

# # Train
# history = model.fit(X_train_processed, y_train_processed, batch_size=64, epochs=20,
#                     validation_data=(X_test_processed, y_test_processed))

# score = model.evaluate(X_test_processed, y_test_processed, verbose=0)

# print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

# plt.figure(figsize=(10, 5))

# plt.subplot(1, 2, 1)
# plt.plot(history.history['val_loss'], color="orange")
# plt.title('Validation loss history')
# plt.ylabel('Loss value')
# plt.xlabel('No. epoch')

# plt.subplot(1, 2, 2)
# plt.plot(history.history['val_accuracy'])
# plt.title('Validation accuracy history')
# plt.ylabel('Accuracy value (%)')
# plt.xlabel('No. epoch')

# plt.tight_layout()
# plt.show()



# history = model.fit(X_train, y_train,
#           batch_size=batch_size,
#           epochs=600,
#           verbose=1,
#           callbacks=[early_stop,tensorboard_callback],
#           validation_data=(X_test, y_test))

# # save the model
# model.save("best_conv.h5")
# print("Model saved successfully")

# %tensorboard --logdir logs

# score = model.evaluate(X_test, y_test, batch_size=50,
#                                       steps=X_test.shape[0] // 50)

# print('Test loss: %.2f' % score[0])
# print('Test accuracy: %.2f'% score[1])
