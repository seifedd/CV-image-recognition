from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from imutils import paths
import argparse
ap=argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True,help="path to input dataset")
ap.add_argument("-k","--neighbors",type=int, default=1,help="number of nearest neighbor for classification")
ap.add_argument("-j","--jobs",type=int,default=-1,help="# of jobs for knn distance (-1 uses all avaialble cores)")
args=vars(ap.parse_args())



print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the image preprocessor, load the dataset from disk,
# and reshape the data matrix
sp = SimplePreprocessor.SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader.SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 3072))

# show some information on memory consumption of the images
print("[INFO] features matrix: {:.1f}MB".format(data.nbytes / (1024 * 1000.0)))


le=LabelEncoder()
labels=le.fit_transform(labels)
(train_X,test_X,train_Y,test_Y)=train_test_split(data,labels,test_size=0.25,random_state=42)


##### K Nearest neighbors (k-NN)#####
print("[INFO] evaluating k-NN classifier...")
model=KNeighborsClassifier(n_neighbors=args["neighbors"],n_jobs=args["jobs"])
model.fit(train_X, train_Y)
print(classification_report(test_Y, model.predict(test_X),
target_names=le.classes_))