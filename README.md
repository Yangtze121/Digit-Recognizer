# Digit-Recognizer
kaggle competition Digit Recognizer
Competition: https://www.kaggle.com/c/digit-recognizer
</br>Data: https://www.kaggle.com/c/digit-recognizer/data

<h2>Competition Description</h2>
</br>MNIST ("Modified National Institute of Standards and Technology") is the de facto “hello world” dataset of computer vision. Since its release in 1999, this classic dataset of handwritten images has served as the basis for benchmarking classification algorithms. As new machine learning techniques emerge, MNIST remains a reliable resource for researchers and learners alike.
</br>In this competition, your goal is to correctly identify digits from a dataset of tens of thousands of handwritten images. We’ve curated a set of tutorial-style kernels which cover everything from regression to neural networks. We encourage you to experiment with different algorithms to learn first-hand what works well and how techniques compare.

<h2>Solution</h2>
7 models are used to predict the dataset, including 6 common machine learning models and a deep learning model.
<h3>1、LGB(LightGBM)</h3>
LGB does not need to deal with the data set too much. It directly feeds the original features into the model, and pays attention to dividing training machines, cross validation sets and test sets when using.
The target is set into multiple classifications, and the optimization index is classified error rate (multi_error).
The training time is 231s, the accuracy is 97.185%, and the accuracy is the highest in the non deep learning model. It's a powerful model.

<h3>2、NB(Naive Bayes)</h3>
The input of the NB model must be a non negative number, and the feature should be scaled to [0,1].
Training and prediction take a very short time and the accuracy is 83.328%.

<h3>3、KNN(K-Neighbors)</h3>
It's a typical lazy learning algorithm and it does not need to deal with the training set.The training time is very short but it takes too long in prediction.
The near neighbor number is set to 5, and the final accuracy is 93.700%.

<h3>4、SVM(Surport Vector Machine)</h3>
SVM model can significantly improve training speed after data standardization. Kernel function uses linear kernel. 
For multi classification problems, the one vs rest principle is used and the voting method is used to predict the results.
The final training time 198s, the accuracy is 91.228%

<h3>5、RF(Random Forest)</h3>
Based on the bagging model, the RF model adds perturbation to the attribute partition, and the ensemble performance is better.
Evaluation index using out of package estimate and the training time is 66s, the accuracy is 94.144%

<h3>6、NN(Neural Network)</h3>
From this opportunity, I have learnt the Tensorflow framework preliminarily, and wrote the NN and CNN models with this framework.
Using 1 hidden layers, the optimization index is cross_entropy, the optimizer uses MomentumOptimizer, 
each 100 sample is divided into one batch, and after running 30 epoch, 3602s is used, the accuracy is 92.357%.

<h3>7、CNN(Convolutional Neural Network)</h3>
CNN is composed of two convolution layers and two full-conneted layers. The optimization index is cross_entropy. 
The optimizer uses AdamOptimizer and runs 30 epoch after 15600s. The accuracy is 98.785%.

<h2>Summary</h2>
As a deep learning algorithm, the CNN model should have the best performance in the training set. In all non depth learning models, 
LGB has the highest performance, moderate training time and high accuracy.
<br/>In the later stage, data pretreatment should be strengthened, and dimension reduction data can be used to improve efficiency.


