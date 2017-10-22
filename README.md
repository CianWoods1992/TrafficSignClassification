# TrafficSignClassification Writeup

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

	Maybe experiment with this a bit before submitting anything
	
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

	- training set accuracy of ?
	- validation set accuracy of ?
	- test set accuracy of ?
	
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
| Stop          | Construction |  0.20       |
| Speed 30      | Construction |  0.20       |
| Construction  | Construction |  0.20       |
| Yield         | Construction |  0.20       |
| Speed 60      | Construction |  0.20       |


Probability and distribution images for each classification

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction.
##### Image 1

##### Image 2

##### Image 3

##### Image 4

##### Image 5
