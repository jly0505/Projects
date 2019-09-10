# CAPSTONE - APPLYING DEEP LEARNING FOR IMAGE RECOGNITION

## Models using FNN, CNN, and VGGNet16


## Table of contents:
* [Executive Summary](#executive-summary)
* [Project Scope](#project-scope)
* [Technology](#technologies)
* [Data Collection](#data-collection)
* [Model Selection Process](#model-selection-process)
* [Model Construction](#model-construction)
* [Conclusion](#conclusion)





## Executive Summary


In all of my travels over the years, I have always been amazed and captivated by the many architectural structures that defy the imaginations. Many of these architectural marvels are built for places of worships, like churches, mosques and temples. Some are majestic like the Notre Dame, the Westminster Abbey, Sensoji Temple in Japan, the Tian Tan Buddha and the Po Lin Monastery in Hong Kong. But many more are small and local. For example, there are over 77,000 temples just in Japan, and over 2,800 in Tokyo alone (1). I've only visited a handful. But sometimes, I don't know which temple I have visited and going through my own photos I cannot clearly differentiate them all. This is what inspired me to do this capstone.

Temples comes in all shapes and sizes, from majestic ones to a simple stone building. The same can be said about churches and mosques. But often times, it is hard to distinguish among them, particularly between a churches and a mosque. For instance, the Hagia Sophia in Istanbul was originally built as an Orthodox Christian Church. And for a thousand years, it remained so until an invasion by an Ottoman Sultan Mehmed in 1453 that promptly turned it into a mosque. Today, it's a museum, but the interior and exterior has features of a church and a mosque. (2)

My motivation to clarify this phenomenon is to use machine learning to capture an image and train it to distinguish its classification into a church, a mosque, or a temple. I will be using deep learning neural networks as the modeling technique to train a model to make predictions based on images. I will test out various models to determine which one will provide the most accurate predictions.

I gather images from google.com image library for three classes, church, mosque and temple. After visually cleaning the dataset I have two balanced classes, church and mosques with roughly 700+ images and a smaller dataset of 394 temples. After some trial and error to identify the most appropriate model to use, I determined a CNN model using transfer learning architect like the VGGNet16, was the best model to use for this image recognition project. I test two models, one is a straightforward VGGNet model (Model 1) and the other is similar but a more densely packed model (Model 2). For Model 1, I was able to capture a 57.5% accuracy score, and for Model 2, a 56.7% accuracy score. These scores are low for most image recognition problems, especially in light of these advance modelling techniques. The main reason for this lower than expected accuracy score is that the data images used are not the best quality images. To improve on the process, better data will need to be gather, perhaps from a different source other than google images. 


(1) http://nbakki.hatenablog.com/entry/Number_of_Temples_in_Japan
(2) https://www.istanbulfantasy.com/hagia-sophia/


## Project Scope

The goal of this project is to build a classification model that can be used to distinhuish between churches, mosques and temple. 

This project is two-folds. The first part of the project requires getting data (images) from https://images.google.com/ The second part requires building a neural network model to read in the images, process the images, fit the images into a model to make predictions. 



## Technologies

#### - Programming Language
        - Python 3.7

#### - Imported Libraries
        - request, json, time, re, glob
        - Pandas
        - Numpy
        - Seaborn
        - Matplotlib.pyplot
        - Scikit-Learn
        - Keras
        - google_images_download



## Data-Collection


Data was collected from https://images.google.com/ using the for google_images_download program for three classes, including churches, mosque and temples. 

A total of about 840 images were collected for two classes, church and mosque. Subsequently, each image in each class was manually scrubbed by visual inspection. For the church and mosque classes, only 51 and 110 images were dropped, respectively. Most images were discarded for various reasons, for instance, a picture of a doorway, an archway, picture too far away, ariel views, duplicate pictures. For the temple class, only 720 images were collected, of which only 394 images were used. Most of the dropped images were due to duplication, images were too close-up, images that were mostly of very modern temples located mainly in the US.



## Model Selection Process

Various modelling construction was explored in this process. Starting with using TensorFlow to process images and then to construct the model to make predictions. I learned and coded this model by following youtube video from sentdex. This process was not pursue after encountering difficulties during the procedures to process the images. 

Next, I discovered several models from the dogs vs cats Kaggle contest and this also proved unsuccessful for various reasons. Finally, I found a much more detailed and rich model build by Adrian Rosebrock, titled "Keras Tutorial: How to get started with Keras, Deep Learning, and Python" (https://www.pyimagesearch.com/2018/09/10/keras-tutorial-how-to-get-started-with-keras-deep-learning-and-python/) I used this model technique to build my image recognition prediction model.



## Model Construction

There are two models I constructed following the instructions from the manual above:
    - A simple FNN model using a 32X32 pixel model with one hidden layer, which I expanded to include a 64X64, and a 128X128 pixel model with multiple hidden layers
    - A CNN model using a tranfer learning architecture, specifically the 64X64 VVGNet16 architect, which I expanded with more hidden layers
    
    
### Preprocessing:

- The first step in the preprocessing is to load the data, then the images are resize to a 64X64 pixels image
- The dataset is then transformed into an array, and a label(y-variable) is create into a seperate array
- The dataset is than normalized (/255)
- The dataset is then train/test/split with a train size: 1,416 images; and test size: 473 images


Other model specifications: Epoch set at 20, Batches at 32, initial learning rate at .01


### FNN models:

The hardest part of the process to build these models were pre-processing the images. The manual provided a method to process the images but I had to adapt the procedures since I did not want to use the command line procedures to set up paths to various directories. Instead, I opted to have my dataset be centrally located within in my python notebook. 

- The first model built was a single layer 32X32 FNN model and the results were not impressive. I got an accuracy score of just 43.5%, just above my baseline of 33%
- In the second model I try to improve it by adding 5 hidden layers, and I got a slightly improved accuracy score at 45.3%
- A third attempt was made to modify this approach with 10 hidden layers and my accuracy score actually went down slightly to 44.6%.
- Next, I decided to improve the pixilation to 64X64, and the model produced a 47.9% accuracy score
- To improve upon the CNN model, I added 5 hidden layers, and regularization techniques such as dropout into the model, with a 46.7% accuracy score

Conclusion: The FNN models are too simplistic and not ideal for image recognition problems. Accuracy scores are below 50%. Interestingly enough, by adding more hidden layers into the model, it did not improve performance.



### CNN models:

Convolutional Neural Network models are much more ideal and is the standard for image recognition problems. In particular, I will be using a transfer learning architect, VGGNet16, to train the model and make predictions. What is the VGGNet16?  The model is a Keras model of 16-layer network that was built by the Visual Geometry Group (VGG) out from Oxford. This team won the ILSVRC-2014 competition sponsored by ImageNet. The ImageNet project is an ongoing effort to collect and classify images. To date, there are over 14.2 million images collect across over 21,800 different classes. 

The VGGNet16 has these parameter: Convolutional layers using only 3X3 dimension, Max pooling layers using only 2X2, Fully Connected (FC) layer at the end and total of 16 layers.

- The first CNN model (Model 1) produced an accuracy score of 57.5%. This model produce 201 errors.

- The second CNN model (Model 2) consist of one additional hidden layer, with a 256X256 pixilation, and produced an accuracy score of 56.7%. This model produced 205 erros.




## Conclusion

#### Model 1: Confusion Matrix:


|        	|        	| Prediction 	| Prediction 	| Prediction 	|
|--------	|:------:	|:----------:	|:----------:	|:----------:	|
|        	|        	|   Church   	|   Mosque   	|   Temple   	|
| Actual 	| Church 	|     98     	|     86     	|     16     	|
| Actual 	| Mosque 	|     45     	|     120    	|     10     	|
| Actual 	| Temple 	|     25     	|     19     	|     54     	|


#### Model 2: Confusion Matrix:


|        	|        	| Prediction 	| Prediction 	| Prediction 	|
|--------	|:------:	|:----------:	|:----------:	|:----------:	|
|        	|        	|   Church   	|   Mosque   	|   Temple   	|
| Actual 	| Church 	|     102    	|     89     	|      9     	|
| Actual 	| Mosque 	|     48     	|     125    	|      2     	|
| Actual 	| Temple 	|     40     	|     17     	|     41     	|



#### Further modeling to consider:

Key Observations:

- The VGGNet16 models take a lot of computing power to run the predcitions. The higher the pixelation, the longer it take to run each eopoch
- I could not run a 256X256 even on a simple FNN model without running out of memory space
- the models tends to converge around 10 - 12 epochs, with the val loss and train loss converging


#### Future work:

- Gather better dataset, maybe explore travel sites, tourism bureau
- Incorporate other pre-trained models like VGG19, Resnet50
- Leverage AWS to get better computing power

