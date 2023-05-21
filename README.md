# AssignmentOne-Keras
CIFAR10 Image Classification with Keras

Q.1 Construct a Convolutional Neural Network for Image Classification using the CIFAR-100 dataset ([https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)) with the following tasks:

1. Increase the no. of layers.
    1. The network should be implemented at a minimum of 4 different sizes
    2. Minimum size of each network must be at least 3 layers
2. Increase the dataset size
    1. Minimum no. of images in each class should be 15
    2. Increment the size of each class by 15 images
    3. The experiment should be conducted at a minimum of 5 different dataset sizes

Dataset:

The CIFAR-100 dataset consists of 100 objects (classes) with 500 images in each class for training and 100 images for testing

Write a report on the **implementation details** of each experimental task. It should also contain the following:

1. Choice of parameters and justification (in your own words) for choosing them
    1. Batch Size
    2. Architecture
    3. Number of Epochs
    4. Learning Rate
    5. Momentum, Decay, Bias (Advanced)
    6. Optimizer
    7. Loss Function
    8. Weight Initialization
    9. Kernel/Filter Size, Stride, Padding
    10. Activation Functions
    11. Size and Number of Fully Connected Layers
    12. Data Preprocessing/Augmentation
    13. Dropout

2. Plot the following graphs
    1. For the largest dataset size
        1. Error-metrics vs. Model size (no. of parameters in millions)
        2. Time taken vs. Model size
    2. For the largest network size
        1. Error-metrics vs. Dataset size
        2. Time taken vs. Dataset size
    3. For the largest dataset size and network size
        1. Error-metrics vs. no. of iterations.
    4. Dataset Size vs Accuracy

3. Insights gained from the experiment conducted such as the behavior of the aforementioned graphs
4. Implementation details:
    1. Network description (layers, feature map dimensions, etc.)
    2. Language and Deep Learning Framework used
    3. Hardware details/specs

Error Metrics:

(i) Validation loss

(ii) Top-5 error

Hyperparameters

Plot Train vs Validation Accuracies &amp; Train vs Validation Loss with Tensorboard

Submission

1. The submission should include the following items in a single .zip folder
  1. A README file with instructions for setting up the environment and executing the source code
  2. Source code
  3. Report

Evaluation Criteria:

The purpose of the assignment is to enable the students to gain a practical understanding of neural networks and for them to gain insights on their behavior. The assignment will be evaluated on the student&#39;s degree of understanding with respect to choice of network parameters and observations on the behavior of the neural network with respect to the size of the model and the dataset.

Bonus:

The following questions may be answered in the report to gain a thorough understanding of Deep Learning concepts. Within the context of the above experiment, answer the following questions:

1. Why is dropout used for regularisation?
2. Which is better: MSE or Cross-entropy? Why?
3. List the dimensions of the feature maps at each layer and without zero-padding.
