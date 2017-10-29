# TrafficSignClassification Writeup

### Data Set Summary & Exploration
#### 1. Provide a basic summary of the data set.
I used python to calculate summary statistics of the traffic sign data set:
* Number of training examples = 34799
* Number of validation examples =  4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.
The following figures show samples taken from the dataset and a chart showing the distribution of the dataset.

![Dataset Sample](/images/dataset_vis.png)
![Dataset Dist](/images/distribution.png)

### Design and Test a Model Architecture
#### 1. Desribe how you preprocessed the data
Before modifying the neural network architecture I decided to preprocess the data to view the impact of preprocessing on a network that I already had a baseline accuracy on. The preprocessing steps I added were as follows, data normalisation, converting image from 3 channel to 1 channel and finally equalising the histogram of the image.


![Preproc Grey](/images/preproc_orig.png)

Firstly I decided to convert the image to grayscale which eliminates different light intensities of the same color to be classified differently.

![Preproc Grey](/images/preproc_grey.png)

Next I equalized the histogram of the image. Histogram equalization improved the contrast of the image and can highlight more features that can be classified on.

![Preproc Hist](/images/preproc_hist.png)

Finally the dataset was normalised for input into the network.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) 
	Consider including a diagram and/or table describing the final model.
	
	Diagram of final network here

| Layer               | Description                                 |               
| ------------------- | ------------------------------------------- |
| Input               | 32X32X1 Grayscale Image                     |
| Flatten             |                                             |
| Spatial Transform   |                                             |
| Convolution         | 1x1 stride, valid padding, outputs 28x28x6  |
| Max Pool            | 2x2 stride, outputs 14x14x6                 |
| Batch Normalisation |                                             |
| Convolution         | 2x2 stride, valid padding, outputs 10x10x32 |
| Max Pool            | 2x2 stride, outputs 5x5x32                  |
| Batch Normalisation |                                             |
| Flatten             | Output 400                                  |
| Fully Connected     | Output 120                                  |
| Dropout             | 0.5 Keep Probability                        |
| Fully Connected     | Output 84                                   |
| Dropout             | 0.5 Keep Probability                        |
| Output              | 43 Classes                                  |
  
#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

In training my model I experimented with the optimiser, batch size and number of epochs. Initially I trained the network using 10 EPOCHS, after adding dropout to my architecture I decided to increase the number of EPOCHS to 30 and 50 in different scenarios to see if the reduction in over fitting for a longer training time could improve the accuracy, this however only increased accuracy on the validation set by 0.08%. The insignificant reduction was not justified in the extra time required to train the network as the laptop running the training had no dedicated GPU. I began training with the Adagrad optimiser, I noticed however that the change in accuracy of validation was in very small increments. I decided to train on the Adam optimiser for comparison. I noticed that the Adam optimiser was converging to its max accuracy in far fewer EPOCHS than Adagrad (~15: Adam, ~40: Adagrad). I then doubled batch size from 128 to 256, this increased accuracy very slightly (~0.05%). To finish I experimented with 30 EPOCHS with the Adam optimiser and a batch size of 128 and for my final training I increased EPOCHS to 100. However, I feel this may have caused slight overfitting on my training set as the validation accuracy and test accuracy reduced by ~4%.
	
#### 4. Describe approach taken...

The initial architecture used was LeNet, mainly as this was the architecture described in the learning material and from initial training it provided a baseline accuracy of 89.4%. Before modifying the neural network architecture I decided to preprocess the data to view the impact of preprocessing on a network that I already had a baseline accuracy on. The preprocessing steps I added were as follows, data normalisation, converting image from 3 channel to 1 channel and finally equalising the histogram of the image. The variance in light sources among images makes it more difficult for a network to pic out important features as the same colors could give different RGB values under different lighting conditions. By converting the images to grayscale and equalising the histogram, the variance in values of pixels under different lighting conditions is minimised making it easier for the network to classify similar features.  

Next I started modifying the LeNet architecture, first adding a dropout layer after the first fully connected layer, I then added a second dropout layer after the second fully connected layer to see if it would add any improvement. The purpose of the dropout layers were to prevent overfitting while training. 

I then added batch normalisation between convolutional layers, the purpose of this is to normalise the data of each batch giving it a zero mean, this increased the overall accuracy of the network.

I then increased the number of nodes in the fully connected layers, hoping for the network to extract more features from the convolution layers. This had little effect.

Finally I added a spatial transform at the beginning of the network, the purpose of this is to transform the input images as they enter the network, this allows the network to find features easier as all of the images are transformed to appear as if they are perpendicular to the camera that took them.
		
My final model results were:

* Training set accuracy of 0.987
* Validation set accuracy of 0.968
* Test set accuracy of 0.941
	
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report.
Here are five German traffic signs that I found on the web:

![Stop](/images/1.jpg)
![Speed 30](/images/2.jpg)
![Construction](/images/3.jpg)
![Yield](/images/4.jpg)
![Speed 60](/images/5.jpg)

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set.
| Image         | Prediciton    | Probability |
| ------------- | ------------- | ----------- |
| Stop          | Priority Road |  0.17       |
| Speed 30      | Speed 30      |  0.91       |
| Construction  | Bumpy Road    |  0.46       |
| Yield         | Yield         |  0.24       |
| Speed 60      | Speed 60      |  1.00       |

##### Probability Distribution
![Probability Distribution](/images/prop_dist.png)

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction.
##### Image 1
| Probability | Prediction    |
| ----------- | ------------- |
| .17         | No Passing    |
| .06         | Speed 80      |
| .06         | Yield         |

##### Image 2

| Probability | Prediction          |
| ----------- | ------------------- |
| .91         | Speed 30            |
| .04         | End of Speed 80     |
| .01         | Speed 20            |

##### Image 3

| Probability | Prediction   |
| ----------- | ------------ |
| .46         | Bumpy Road   |
| .28         | Speed 120    |
| .06         | Keep Right   |

##### Image 4

| Probability | Prediction      |
| ----------- | --------------- |
| .24         | Yield           |
| .12         | Speed 50        |
| .06         | Speed 80        |

##### Image 5

| Probability | Prediction      |
| ----------- | --------------- |
| 1.0         | Speed 60        |
| .00         | Speed 80        |
| .00         | Speed 50        |
