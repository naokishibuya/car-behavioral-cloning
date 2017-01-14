## Project Description

In this project, the neural network is given a task of cloning the car driving behavior.  As such, I conducted a supervised learning on the car steering angle given road images in front of a car.  There are three images from the center, the left and the right angles associated with the car.  It is a supervised regression problem.  As image processing is involved, the convolutional neural network was chosen, especially [the NVIDIA model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) that was proven to work in the same kind of problem domain.


###Files included

- model.py The script used to create and train the model.
- drive.py The script to drive the car. You can feel free to resubmit the original drive.py or make modifications and submit your modified version.
- utils.py The script to provide useful functionalities (i.e. image preprocessing and augumentation)
- model.json The model architecture.
- model.h5 The model weights.


## Model Architecture Design

###What kind of reference have you been studying for?

I’ve studied [the NVIDIA model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) for the end-to-end self driving.

###Why this model is suitable for this question?

The NVIDIA model has been used by NVIDIA for the end-to-end self driving test as such the model is well suited for the project.  The reason behind this is that it is using the convolution layers which works well with supervised image classification / regression problems.  As the model is well documented, I was able to focus how to adjust the training images to produce the best result with some adjustments to the model to avoid overfitting and adding non-linearity to improve the prediction.


###How did you decide the number and type of layers?

The model is based on the NVIDIA architecture.  The NVIDIA model was introduced in the course and it is an proven-to-work solution for behavioral cloning task.  However, I've added the following adjustments to the model. 

- I used Lambda layer to normalized input images to avoid saturation and make gradients work better.
- I've added an additional dropout layer to avoid overfitting after the convolution layers.
- I've also included ELU for activation function for every layer except for the output layer to introduce non-linearity.

In the end, the model looks like as follows:

- Image normalization
- Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
- Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
- Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
- Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
- Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
- Drop out (0.5)
- Fully connected: neurons: 100, activation: ELU
- Fully connected: neurons:  50, activation: ELU
- Fully connected: neurons:  10, activation: ELU
- Fully connected: neurons:   1 (output)

As per the NVIDIA model, the convolution layers are meant to handle feature engineering and the fully connected layer for predicting the steering angle.  However, as stated in the NVIDIA document, it is not clear where to draw such a clear distinction.  Overall, the model is very functional to clone the given steering behavior.  

The below is an model structure output from the Keras which gives more details on the shapes and the number of parameters.

| Layer (type)                   |Output Shape      |Params  |Connected to      |
|--------------------------------|------------------|-------:|------------------|
|lambda_1 (Lambda)               |(None, 66, 200, 3)|0       |lambda_input_1    |            
|convolution2d_1 (Convolution2D) |(None, 31, 98, 24)|1824    |lambda_1          |                   
|convolution2d_2 (Convolution2D) |(None, 14, 47, 36)|21636   |convolution2d_1   |            
|convolution2d_3 (Convolution2D) |(None, 5, 22, 48) |43248   |convolution2d_2   |            
|convolution2d_4 (Convolution2D) |(None, 3, 20, 64) |27712   |convolution2d_3   |            
|convolution2d_5 (Convolution2D) |(None, 1, 18, 64) |36928   |convolution2d_4   |            
|dropout_1 (Dropout)             |(None, 1, 18, 64) |0       |convolution2d_5   |            
|flatten_1 (Flatten)             |(None, 1152)      |0       |dropout_1         |                  
|dense_1 (Dense)                 |(None, 100)       |115300  |flatten_1         |                  
|dense_2 (Dense)                 |(None, 50)        |5050    |dense_1           |                    
|dense_3 (Dense)                 |(None, 10)        |510     |dense_2           |                    
|dense_4 (Dense)                 |(None, 1)         |11      |dense_3           |                    
|                                |**Total params**  |252219  |                  |


## Data Preprocessing

###Image Sizing

- the images are cropped so that the model won’t be trained with the sky and the car front parts
- the images are resized to 66x200 (3 YUV channels) as per NVIDIA model
- the images are normalized (image data divided by 127.5 and subtracted 1.0).  As stated in the Model Architecture section, this is to avoid saturation and make gradients work better)


## Model Training

###Image Augumentation

For training, I used the following augumentation technique along with Python generator to generate unlimited number of images:

- Randomly choose right, left or center images.
- For left image, steering angle is adjusted by +0.2
- For right image, steering angle is adjusted by -0.2
- Randomly flip image left/right
- Randomly translate image horizontally with steering angle adjustment (0.002 per pixel shift)
- Randomly translate image virtically

Using the left/right images is useful to train the recovery driving scenario.  The horizontal translation is useful for difficult curve handling (i.e. the one after the bridge).


### Examples of Augmented Images

The following is the example transformations:

**Center Image**

![Center Image](images/center.png)

**Left Image**

![Left Image](images/left.png)

**Right Image**

![Right Image](images/right.png)

**Flipped Image**

![Flipped Image](images/flip.png)

**Translated Image**

![Translated Image](images/trans.png)


## Training, Validation and Test

I splitted the images into train and validation set in order to measure the performance at every epoch.  Testing was done using the simulator.


###How to evaluate the model

As for training, I used mean squared error for the loss function to measure how close the model predicts to the given steering angle for each image.


I used Adam optimizer for optimization with learning rate of 1.0e-4 which is smaller than the default of 1.0e-3.  The default value was too big and made the validation loss stop improving too soon.

I used ModelCheckpoint from Keras to save the model only if the validation loss is improved which is checked for every epoch.

As there can be unlimited number of images augmented, I set the samples per epoch to 20,000.  I tried from 1 to 200 epochs but I found 5 epochs is good enough to produce a well trained model.  The batch size of 40 was chosen as that is the maximum size which does not cause out of memory error on my Mac.



## Outcome

The model can drive the course without bumping into the side ways.

