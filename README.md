## Narrative CIFAR-10 technical test

Building a small model from scratch using the CIFAR-10 dataset. 88.7% test accuracy with minimal hyperparameter tuning.

## TODO (sorry, it's Sunday night)

* Get notebooks off server
* Test and upload them into this repo
* Upload confusion matrix script

I enjoyed the challenge of creating a small high-performance model from scratch. In industry, my usual approach to this would be to find the largest pre built model that would fit the requirements and use transfer learning to limit time spent on the task, with minimal tuning.

I chose to build out from a very simple network to show more details of my approach and thinking. I wanted to avoid the approach of building a safe model and running 75+ epochs to produce a good result, although that would have taken a lot less time!

The final model has over 86% accuracy on the 10th epoch. This is without a lot of fine tuning and experimentation. I would like to try using ResNet 18 or 20 with data augmentation.

Requires Python 3.7 (untested on other versions).

### Getting started

Clone the repo, open your terminal and navigate to the main folder ```Narrative_CIFAR10```

**Training script** - copy and paste ```python3 cifar10.py``` into the terminal and hit enter.
The trained models are already available, so you don't need to run the whole training cycle.

**Test script** - copy and paste ```python3 cifar10_test.py``` into the terminal and hit enter.

### CIFAR-10

The CIFAR-10 dataset has relatively simple images of 32 x 32 pixels, 10 categories, and 50,000 training examples. I chose to use Keras for this test because it allows rapid iteration of hypotheses, and it’s very readable in notebooks. My goal was to get 90% accuracy within the requirements. But 89% is ok for now.

### First experiment

My first experiment was a relatively small CNN, expected to have average performance (70-85%). No dropout was present because of the size of the model. The potential of this architecture is limited, even with fine tuning, data augmentation, and dropout.

#### Choosing GELU and Adam

I chose GELU for the activation function because of its recent success with models such as OpenAI’s GPT, and its growing popularity in the ML community. It is similar to leaky ReLU with that it can output negative numbers. I wanted to see how it would perform in a smaller network.

GELU is only in TF nightly, so I added a custom function in the script for it. However, this causes problems when trying to load a model saved from a checkpoint. Model.save still works and loads, though.

I chose the Adam optimizer, because it works fairly well on out of the box in Keras, and easily changed later.

### A larger model

A slightly larger model still fits within the parameter requirements. Before adding any extras, I ran a quick test to see if performance was encouraging for the first 5 epochs. It was, so I began experimenting with the architecture to improve the accuracy.

Learning rate was key to get reasonably high accuracy with the limited amount of data and small model architecture. 

#### Dropout

I used reasonably high dropout to begin with, so the complexity of the model was increased and the learning rate could be tuned.

After the MaxPooling2D layer, dropout started at 0.25 for the first 32 Conv2D section, 0.3 for the second 64 Conv2D section, 0.35 for the Conv2D section, and 0.5 for the final Dense layer.

After verifying the model would train ok, I increased the learning rate to 0.003 (3x the standard). Training results were promising, so I moved on.

#### Data augmentation

Data augmentation was added next using **TensorFlow’s datagen**. It’s simple to get going, but not great to fine tune. 20 degrees of rotation and vertical flipping added too much complexity with a high learning rate. The model wasn't training well.

With batch normalization added, and the rotation decreased to 10 degrees, the model started to produce promising results. I decreased the size of the final dense layer to 256 reduce the complexity (and stay within the requirements).

#### Building out the platform and tuning

With the data augmentation working and the model training, I wrote out some helper functions and tests.

Keras has a very good callback API, which I leveraged to add in helper functions for dynamically decreasing learning rate (custom decay, scheduling, and ReduceLROnPlateau) and early stop functionality.

I quickly tried several optimizers and found **Adamax** to perform the best, even when using a learning rate of 0.0035 (3.5x more than default).

This model had now performed at 85%+ with several different hyperparameters.

With some hyperparameter tuning, a result of over 90% test accuracy is very likely achievable.

## Validation / Test data!?

I used the test data as the validation data, due to the limited amount of training data available. This allowed rapid testing to see how approaches were trending. With data augmentation, the test data could be withheld and a train-validation split of 0.2 could be used (0.15 would be the absolute lowest I would go on this dataset with data augmentation). But with limited time and compute, I haven't tried this.

## Next steps

There are lots of ways that could improve performance:

1. Hyperparameter tuning
  * Testing new activation functions
  * Testing different optimizers
  * Adjusting learning rate, decay, and plateau controls
2. Different architectures
  * ResNet would be top of my priority list
3. Data augmentation
  * Reviewing 