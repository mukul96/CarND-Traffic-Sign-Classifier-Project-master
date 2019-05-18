# **Traffic Sign Recognition**


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"


#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data 

![images explored](/output_images/explore.png)

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because color channels are not really required to train the model and they add extra confusion during training

Here is an example of a traffic sign image before and after grayscaling.


![images explored](/output_images/grayscale.png)

As a last step, I normalized the image data because it will speed up the training and also due to the non clarity of the images it would not be possible to do it in RGB.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		|   32x32x1 Grayscale image						| 
| Convolution 3x3     	|   32x32x1 -> 28x28x12 	                    |
| RELU					|												|
| Max pooling	      	|   28x28x12 -> 14x14x12                 	    |
| Convolution 3x3     	|   14x14x12 -> 10x10x25                        |
| RELU					|												|
| Max pooling	      	|   10x10x25 -> 5x5x25				            |
| Flatten				|	5x5x25   -> 625						        |		
| Drop out: 			|   625      -> 625	                            |
| Fully connected       |   625      -> 300                             |
| RELU					|												|
| Fully connected       |   300      -> 150                             |
| RELU					|												|
| Fully connected       |   150      -> 43                              |



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used LeNet architecture. I used the AdamOptimizer with a learning rate of 0.001. The epochs used was 15 with the batch size also 60. This gave me an accuracy of 93.7% on validation set and 92.53% accuracy on test set

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.5%
* validation set accuracy of 93-94%
* test set accuracy of 92-93%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
    I used LeNet architecture with some modification by adding layers and removing also so that I could build the model myself which initially gave me validation accuracy of around 91% but I increased this accuracy by tweaking epoch value from 10 to 15 and batch size from 64 to 60. The reason of increasing the epoch value to 50 was that due to the underfitting of data initially I thought the model was not trained properly
* What were some problems with the initial architecture?
    The major problem was the confusion of which optimizer to use so that the results were better.

* Which parameters were tuned? How were they adjusted and why?
    There is nothing called paramters to tune in DL therer are hyper paramters which were tuned in order to properly fit the data and the strides were also changed accordingly due to the image size.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
    Accuracy of 93.6% on test set gave me the confidence to test this model on new images taken from web. 4 out of 5 such images were predicted correctly using this model which proved that this model is working model. Although I ll work further on improving this model to get better accuracy.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:
![first](/my_images/1.png)
![second](/my_images/2.png)
![third](/my_images/3.jpg)
![fourth](/my_images/4.png)
![fifth](/my_images/5.jpg)


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

You can check the detailed probabilities of the image in the traffic sign classifier notebook under the Analyze 
