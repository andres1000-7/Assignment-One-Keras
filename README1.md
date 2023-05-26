# AssignmentOne-Keras

## CIFAR100 Image Classification with Keras

### Setting up the environment
All of the source code was mainly edited in PyCharm and executed on [https://colab.research.google.com/](https://colab.research.google.com/). To setup the environment in colab, no additional installations are needed, one must only import the necessary tensorflow and keras libraries to execute the source code. However, to edit the source code on PyCharm, the following steps were performed: 
 1. Anaconda installation for Windows was followed in
    [https://docs.anaconda.com/anaconda/install/windows/](https://docs.anaconda.com/anaconda/install/windows/)
 2. Run the command on Anaconda Prompt to create the virtual environment: `conda create -n reu2023 python=3.10 anaconda`
 3. Run `conda install tensorflow` and `conda install -c conda-forge keras` to intall the packages.

### Executing the source code
The source code is a single .py file that performs the assignment tasks. To execute the source code, the following must be followed:

 1. Lines 1 to 95 define the overall structure of the program, where the cifar100 data is loaded and functions are defined for increasing the dataset, preprocessing data, and creating new models. This goes along with some important variables and objects such as num_layer_list, dataset_sizes, etc. These lines cannot be commented.
 2. Lines 129 to 209 accomplish the following task: For the largest dataset size:
	 a. Error-metrics vs. Model size (no. of parameters in millions)
	 b. Time taken vs. Model size
	 These lines can be uncommented to only run this task.	 
 3. Lines 211 to 285 accomplish the following task: For the largest network size
    a.  Error-metrics vs. Dataset size
    b.  Time taken vs. Dataset size
    These lines can be uncommented to only run this task.	 
        
 4. Lines 287 to 326 accomplish the task: For the largest dataset size and network size
 a. Error-metrics vs. no. of iterations
 b. Dataset Size vs Accuracy
 
 5. Lines 328 to 366 do the following: Plot Train vs Validation Accuracies & Train vs Validation Loss with Tensorboard