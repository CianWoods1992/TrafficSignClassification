# TrafficSignClassification Writeup

### Data Set Summary & Exploration
#### 1. Provide a basic summary of the data set.
I used python to calculate summary statistics of the traffic sign data set:
..* Number of training examples = 34799
..* Number of validation examples =  4410
..* Number of testing examples = 12630
..* Image data shape = (32, 32, 3)
..* Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.
The following figures show samples taken from the dataset and a chart showing the distribution of the dataset.

![Dataset Sample](/images/dataset_vis.png)
![Dataset Dist](/images/distribution.png)

### Design and Test a Model Architecture
#### 1. Desribe how you preprocessed the data
Grayscale before and after

equaliseHist before and after

reshape before and after

normalise before and after

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) 
	Consider including a diagram and/or table describing the final model.
	
	Diagram of final network here

| Layer               | Description   |               
| ------------------- | ------------- |
| Input               | 32X32X1 Image |
| Flatten             |               |
| Spatial Transform   |               |
| Convolution         |               |
| Max Pool            |               |
| Batch Normalisation |               |
| Convolution         |               |
| Max Pool            |               |
| Batch Normalisation |               |
| Flatten             |               |
| Fully Connected     |               |
| Dropout             |               |
| Fully Connected     |               |
| Dropout             |               |
| Output              |               |
  
#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

In training my model I experimented with the optimiser, batch size and number of epochs. Initially I trained the network using 10 EPOCHS, after adding dropout to my architecture I decided to increase the number of EPOCHS to 30 and 50 in different scenarios to see if the reduction in over fitting for a longer training time could improve the accuracy, this however only increased accuracy on the validation set by 0.08%. The insignificant reduction was not justified in the extra time required to train the network as the laptop running the training had no dedicated GPU. I began training with the Adagrad optimiser, I noticed however that the change in accuracy of validation was in very small increments. I decided to train on the Adam optimiser for comparison. I noticed that the Adam optimiser was converging to its max accuracy in far fewer EPOCHS than Adagrad (~15: Adam, ~40: Adagrad). I then doubled batch size from 128 to 256, this increased accuracy very slightly (~0.05%). To finish I experimented with 30 EPOCHS with the Adam optimiser and a batch size of 128 and for my final training I increased EPOCHS to 100. However, I feel this may have caused slight overfitting on my training set as the validation accuracy and test accuracy reduced by ~4%.
	
#### 4. Describe approach taken...

	1. LeNet (3 channel, no preprocessing, 10E) - 89.4%
	2. LeNet (3 channel, normalisation, 10E) - 87.3%
	3. LeNet (1 channel, normalisation, 10E) - 86.7 %
	4. LeNet with Dropout between FC1 and FC2 (1 channel, normalise hist) - 88.7%
	5. LeNet with Dropout after FC1 and after FC2 (1 channel, normalise hist) - 93.8%
	5. Added batch normalisation - 96.9%
	6. Added spatial transform - 97.2%
	7. Added L2 Regularisation
		tried sobel... why?
		
My final model results were:

	- training set accuracy of 0.994
	- validation set accuracy of 0.973
	- test set accuracy of 0.954
	
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Stop](/images/1.jpg)
![Speed 30](/images/2.jpg)
![Construction](/images/3.jpg)
![Yield](/images/4.jpg)
![Speed 60](/images/5.jpg)

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set.
| Image         | Prediciton   | Probability |
| ------------- | ------------ | ----------- |
| Stop          | No Passing   |  0.39       |
| Speed 30      | Speed 30     |  0.88       |
| Construction  | Bumpy Road   |  0.71       |
| Yield         | Yield        |  0.67       |
| Speed 60      | Speed 60     |  0.48       |

Predictions on the test set...

##### Probability Distribution
![Probability Distribution](/images/prob_dist.png)

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction.
##### Image 1
| Probability | Prediction   |
| ----------- | -----------  |
| .08         | Construction |
| .06         | Stop Sign    |
| .02         | Yield        |

##### Image 2

| Probability | Prediction   |
| ----------- | -----------  |
| .08         | Construction |
| .06         | Stop Sign    |
| .02         | Yield        |

##### Image 3

| Probability | Prediction   |
| ----------- | -----------  |
| .08         | Construction |
| .06         | Stop Sign    |
| .02         | Yield        |

##### Image 4

| Probability | Prediction   |
| ----------- | -----------  |
| .08         | Construction |
| .06         | Stop Sign    |
| .02         | Yield        |

##### Image 5

| Probability | Prediction   |
| ----------- | -----------  |
| .08         | Construction |
| .06         | Stop Sign    |
| .02         | Yield        |
