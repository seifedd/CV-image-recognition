Disclaimer: I don't own this code, I just have used the code in this book (Please buy from [pyimagesearch.com]{https://www.pyimagesearch.com/deep-learning-computer-vision-python-book/}) , and added to it few pieces to build my image classifier. 
Book: Deep Learning for Computer Vision with Python – Starter Bundle written by Dr. Adrian Rosebrock.

## This folder consists of a building block for an image classifier, which can be used independently of the :

1.  class SimpleDatasetLoader: Loads the images, labels into a lists and returns a numpy array for both the images and labels. Uses CV2. 

2.  class SimplePreprocessor: This is for re-sizing the images as we all know, before training the models, we need to be cognizant of our  dataset size before even starting to work with image classification algorithms. 
3. Finally, the image classification, using as arguments 1) the path to input dataset" 2)number of nearest neighbor classification 3) # of jobs for KNN distance (-1 uses all avaialble cores)
   
# CV-image-recognition

We use sklearn library for image classification, in this case, and as ML classifier we use k-Nearest Neighbors
(k-NN) classifier. 

## Dataset 

Images inside the Animals dataset belong to three distinct classes: dogs, cats, and pandas,
with 1,000 example images per class. The dog and cat images were sampled from the Kaggle
Dogs vs. Cats challenge (http://pyimg.co/ogx37). . Best of all, a deep
learning model can quickly be trained on this dataset on either a CPU or GPU. Regardless of your
hardware setup, you can use this dataset to learn the basics of machine learning and deep learning.

## Results obtained:

`Our goal in this chapter is to leverage the k-NN classifier to attempt to recognize each of these
species in an image using only the raw pixel intensities (i.e., no feature extraction is taking place).
As we’ll see, raw pixel intensities do not lend themselves well to the k-NN algorithm. Nonetheless,
this is an important benchmark experiment to run so we can appreciate why Convolutional Neural
Networks are able to obtain such high accuracy on raw pixel intensities while traditional machine
learning algorithms fail to do so.`
