Disclaimer: I don't own this code, I just have used the code in this book (Please buy from [pyimagesearch.com](https://www.pyimagesearch.com/deep-learning-computer-vision-python-book/)) for learning purposes, and added to it few pieces to build my image classifier (For learning purposes). 

## Book: 

Deep Learning for Computer Vision with Python – Starter Bundle written by Dr. Adrian Rosebrock.

## This folder consists of a building block for an image classifier from pyimagesearch.com
1.  class SimpleDatasetLoader: Loads the images, labels into a lists and returns a numpy array for both the images and labels. Uses CV2. 

2.  class SimplePreprocessor: This is for re-sizing the images, as we all know before training the models, we need to be cognizant of our dataset size before even starting to work with image classification algorithms. 
3. Finally, the image classification, using as arguments 1) the path to input dataset" 2)number of nearest neighbor classification 3) # of jobs for KNN distance (-1 uses all avaialble cores)
> 1. and 2. Can be used independently of the training model, which makes this code reausable. 
# CV-image-recognition

We use sklearn library for image classification, and as ML classifier we use k-Nearest Neighbors (k-NN) classifier. 

## Dataset 

Images inside the Animals dataset belong to three distinct classes: dogs, cats, and pandas,
with 1,000 example images per class. The dog and cat images were sampled from the Kaggle
Dogs vs. Cats challenge (http://pyimg.co/ogx37). . Best of all, a deep
learning model can quickly be trained on this dataset on either a CPU or GPU. Regardless of your
hardware setup, you can use this dataset to learn the basics of machine learning and deep learning.

Note: Our feature matrix only consumes 9MB of memory for 3,000 images, each of size
32×32×3 – this dataset can easily be stored in memory on modern machines without a problem.
## Results obtained:

`Our goal in this chapter is to leverage the k-NN classifier to attempt to recognize each of these
species in an image using only the raw pixel intensities (i.e., no feature extraction is taking place).
As we’ll see, raw pixel intensities do not lend themselves well to the k-NN algorithm. Nonetheless,
this is an important benchmark experiment to run so we can appreciate why Convolutional Neural
Networks are able to obtain such high accuracy on raw pixel intensities while traditional machine
learning algorithms fail to do so.`

Evaluating our classifier, we see that we obtained 52% accuracy – this accuracy isn’t bad for a
classifier that doesn’t do any true “learning” at all, given that the probability of randomly guessing
the correct answer is 1/3.
However, it is interesting to inspect the accuracy for each of the class labels. The “panda” class
was correctly classified 79% of the time, likely due to the fact that pandas are largely black and
white and thus these images lie closer together in our 3,072-dim space.
Dogs and cats obtain substantially lower classification accuracy at 39% and 36%, respectively.
These results can be attributed to the fact that dogs and cats can have very similar shades of fur
coats and the color of their coats cannot be used to discriminate between them. Background noise
(such as grass in a backyard, the color of a couch an animal is resting on, etc.) can also “confuse”
the k-NN algorithm as its unable to learn any discriminating patterns between these species. This
confusion is one of the primary drawbacks of the k-NN algorithm: while it’s simple, it is also
unable to learn from the data.
