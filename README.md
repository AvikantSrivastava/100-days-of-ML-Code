# 100 Days Of ML Code

## Day 1 : March 30, 2020 _MNIST GAN_

<h3><div style="text-align:center">Generated Fake handwritten digits using AI</div></h3>
<div style="text-align:center"><img src= "https://raw.githubusercontent.com/AvikantSrivastava/100-days-of-ML-Code/master/Day1%20%3A%20MNIST%20GAN/dcgan.gif"></div>

**Today's Progress:** Today I worked on DCGAN (Deep Convolutional Generative Adversarial Network). Implemented on the MNIST handwritten digits dataset to generate Handwritten digits.

**Thoughts:** I will try DCGAN on slightly more complex datasets such as Fashion MNIST and CIFAR-10.

**Link of Work:** 
- [ipynb](https://github.com/AvikantSrivastava/100-days-of-ML-Code/blob/master/Day1%20:%20MNIST%20GAN/mnist_gan.ipynb)

**References**

- [DCGANs (Deep Convolutional Generative Adversarial Networks)](https://towardsdatascience.com/dcgans-deep-convolutional-generative-adversarial-networks-c7f392c2c8f8)
- [Deep Convolutional Generative Adversarial Network TensorFlow Core](https://www.tensorflow.org/tutorials/generative/dcgan)

## Day 2: March 31, 2020 _CIFAR-10_

<h3><div style="text-align:center">Basic Image Classification using AI</div></h3>
<div style="text-align:center"><img src= "https://github.com/AvikantSrivastava/100-days-of-ML-Code/raw/master/Day2%20:%20CIFAR-10/bird.png"></div>
<div style="text-align:center"><img src= "https://github.com/AvikantSrivastava/100-days-of-ML-Code/raw/master/Day2%20:%20CIFAR-10/car.png"></div>
<div style="text-align:center"><img src= "https://github.com/AvikantSrivastava/100-days-of-ML-Code/raw/master/Day2%20:%20CIFAR-10/plane.png"></div>


**Today's Progress:** Built a network for CIFAR-10 dataset comprising Convolution, Max Pooling, Batch- Normalization and Dropout layers.
Studied about dropout and batch normalization in detail and various ways to avoid overfitting in a network.

**Thoughts:** Looking forward to tweak the model and obtain a better accuracy and reduce the number of parameters.

**Link of Work:**


- [ipynb](https://github.com/AvikantSrivastava/100-days-of-ML-Code/blob/master/Day2%20:%20CIFAR-10/cifar10.ipynb)
- [test.py](https://github.com/AvikantSrivastava/100-days-of-ML-Code/blob/master/Day2%20:%20CIFAR-10/test1.py)

**References**

- [Achieving 90% accuracy in Object Recognition Task on CIFAR-10 Dataset with Keras: Convolutional Neural Networks - Machine Learning in Action ](https://appliedmachinelearning.blog/2018/03/24/achieving-90-accuracy-in-object-recognition-task-on-cifar-10-dataset-with-keras-convolutional-neural-networks/)
- [CIFAR-10 Image Classification in TensorFlow - Towards Data Science ](https://towardsdatascience.com/cifar-10-image-classification-in-tensorflow-5b501f7dc77c)

## Day 3 : April 1, 2020 _Transfer Learning with InceptionV3_

**Today's Progress:** Today my goal was to get started with transfer learning and understand the architecture. I chose [InceptionV3](https://en.wikipedia.org/wiki/Inceptionv3) begin with. Started building the model by picking weights from Imagenet on Inception as the base model. Added a couple of dense layer with dropouts to complete the model.
The model was trained on 'Cats vs Dogs' dataset for 10 epochs with a training accuracy of 98.69%.

**Thoughts:** I am looking forward to implement the concept of Transfer Learning on a more project with a gain of accuracy.
Also I am excited to try out more architectures such as ResNet, VGG and AlexNet. 

**Link of Work:**
- [ipynb](https://github.com/AvikantSrivastava/100-days-of-ML-Code/blob/master/Day3:%20Transfer%20Learning%20with%20InceptionV3/cats_vs_dogs_inceptionv3.ipynb)

**References**

- [Master Transfer learning by using Pre-trained Models in Deep Learning	](https://www.analyticsvidhya.com/blog/2017/06/transfer-learning-the-art-of-fine-tuning-a-pre-trained-model/)
- [tf.keras.applications.InceptionV3 TensorFlow Core v2.1.0](https://www.tensorflow.org/api_docs/python/tf/keras/applications/InceptionV3)

- [Using Inception-v3 from TensorFlow Hub for transfer learning ](https://medium.com/@utsumuki_neko/using-inception-v3-from-tensorflow-hub-for-transfer-learning-a931ff884526)

## Day 4 : April 2 , 2020 *Mask RCNN*

**Today's Progress:**  Today I tried to understand the idea behind Mask Region-based Convolution Neural Network better known as Mask RCNN. While going though the references I also learned the following things
- Object Localization
- Instance Segmentation
- Semantic Segmentation
- ROI Pooling

**References**

- [GitHub - matterport/Mask_RCNN: Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow](https://github.com/matterport/Mask_RCNN)
  
- [Mask R-CNN with OpenCV - PyImageSearch](https://www.pyimagesearch.com/2018/11/19/mask-r-cnn-with-opencv/)

- [Review: DeepMask (Instance Segmentation) - Towards Data Science](https://towardsdatascience.com/review-deepmask-instance-segmentation-30327a072339)

## Day 5 : April 3 , 2020 *Mask RCNN on Images*

**Today's Progress:** Today I implemented Mask RCNN on Images. I used Open CV as the platform to work. The model which I took for this task was trained on InceptionV2 on the COCO Dataset. 

**Thoughts:** I am planning to implement Mask-RCNN next on videos. I want to work on the challenges with the video and learn about video processing all together.

**Link of Work:** 
- [mask_rcnn.py](https://github.com/AvikantSrivastava/100-days-of-ML-Code/blob/master/Day4%20and%205%20:%20Mask%20RCNN/mask_rcnn.py)

**References**

- [Mask R-CNN with OpenCV - PyImageSearch](https://www.pyimagesearch.com/2018/11/19/mask-r-cnn-with-opencv/)


## Day 6 : April 4 , *Mask RCNN on Videos*

**Today's Progress:** Continuing with yesterday's work, I implemented Mask RCNN on video feed. The project was based on the same architecture and dataset as yesterday's. I tweaked the script to work on videos.

**Thoughts:** Today's implementation was quite computationally expensive. A 120-frame, 4-second video took around 10 minutes to process. The network may not be the fastest but it is quite good in terms of accuracy of detecting and masking objects. So I want to try out more computer vision techniques to do same or a similar job.

**Link of Work:** 
- [mask_rcnn_videos.py](https://github.com/AvikantSrivastava/100-days-of-ML-Code/blob/master/Day6%20:%20Mask-RCNN%20on%20Videos/mask_rcnn_videos.py)

**References**

- [Mask R-CNN with OpenCV - PyImageSearch](https://www.pyimagesearch.com/2018/11/19/mask-r-cnn-with-opencv/)



## Day 7 : April 5 , *Object Detection using Deep Learning*

**Today's Progress:** Today I dove deep into the most in-demand application of the deep learning ie. Object Detection. So I started reading about the various existing architectures. 
- Hog Features
- R-CNN
- Spatial Pyramid Pooling(SPP-net)
- Fast R-CNN
- Faster R-CNN
- YOLO(You only Look Once)
- Single Shot Detector(SSD)

I discovered the working of these sophisticated architectures and compared the output result.

**Thoughts:** After reading about such networks 'YOLO' and 'SSD' intrigued me the most. So I am looking forward to implement those network in a project form on images and videos.

**References**
- [Zero to Hero: Guide to Object Detection using Deep Learning: Faster R-CNN,YOLO,SSD](https://cv-tricks.com/object-detection/faster-r-cnn-yolo-ssd/)
- [A 2019 Guide to Object Detection - Heartbeat	](https://heartbeat.fritz.ai/a-2019-guide-to-object-detection-9509987954c3)


## Day 8 : April 6, *Deep Generative Models*

**Today's Progress:** Today I started with Deep Generative Modeling as part of [MIT's Introduction to Deep Learning](http://introtodeeplearning.com/).

**References**
- [Deep Generative Models - Towards Data Science](https://towardsdatascience.com/deep-generative-models-25ab2821afd3)
- [Deep Generative Modeling MIT 6.S191 Youtube](https://www.youtube.com/watch?v=rZufA635dq4)
- [Deep Generative Modeling Slides](http://introtodeeplearning.com/slides/6S191_MIT_DeepLearning_L4.pdf)


## Day 9 : April 7, *Linear Regression in Numpy*

**Today's Progress:** Today I read about learning regression in detail with the implementation in numpy. I used normal equation to calculate the weights of a function. The weights were determined on a random generated data. The data contained x and y pair with a linear relation of ' y = 4 + 3x'.

**Thoughts:** Understanding the algorithms at the fundamental level is a requisite for anyone who practices Machine Learning. Looking forward to understand the basic methods at a fundamental level.

**Link of Work:**

-  [[code]](https://github.com/AvikantSrivastava/100-days-of-ML-Code/blob/master/Day9%20:%20Linear%20Regression/linear_regression.md)
- ![](https://raw.githubusercontent.com/AvikantSrivastava/100-days-of-ML-Code/master/Day9%20%3A%20Linear%20Regression/output_8_0.png)

**References** 
- [Hands–On Machine Learning with Scikit–Learn and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)


## Day 10 : April 8, *Gradient Descent techniques*

**Today's Progress:** I learned about the various types of gradient descent methods namely Batch Gradient Descent,Stochastic Gradient Descent and Mini Batch Gradient Descent. I implemented the same with sklearn library in python. I also learned about learning rate schedule and made a LR schedular.
And finally compared the speed, architecture and use of various Gradient Descent techniques. 

**Thoughts:** It is good to know, what goes on underneath every process in Machine Learning.

**Link of Work:**

- [[code]](https://github.com/AvikantSrivastava/100-days-of-ML-Code/blob/master/Day10:%20Gradient%20Descent/gradient%20descent.md)

**References** 
- [Hands–On Machine Learning with Scikit–Learn and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)


## Day 11 : April 9, *Polynomial Regression*

**Today's Progress:** Worked on polynomial regression to fit the curves with higher degree. Made a simple dataset(collection on random points which falls near the equation) and analysed it with polynomial regression.

**Link of Work:**

-   [[code]](https://github.com/AvikantSrivastava/100-days-of-ML-Code/blob/master/Day11%20:%20Polynomial%20Regression/polynomical%20regression.md)
- - Blue: Train Accuracy
  - Red : Test Accuracy
  - ![](https://raw.githubusercontent.com/AvikantSrivastava/100-days-of-ML-Code/master/Day11%20%3A%20Polynomial%20Regression/output_7_0.png)
- 10 degree polynomial
- - ![](https://raw.githubusercontent.com/AvikantSrivastava/100-days-of-ML-Code/master/Day11%20%3A%20Polynomial%20Regression/output_10_0.png)

**References** 
- [Hands–On Machine Learning with Scikit–Learn and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)


## Day 12 : April 10, *YOLO object detection*

**Today's Progress:** I worked on Yolo and implemented it in OpenCV. I tried two different versions of weights from Darknet yolov3-tiny.weights and yolov3.weights.

**Thoughts:** Yolo is fast compared to other algorithms and can be implemented on hardware constraints environments without a sweat.

**Link of Work:**
- [[code]](https://github.com/AvikantSrivastava/100-days-of-ML-Code/blob/master/Day12%20:%20Yolo/yolo.py)
![](https://github.com/AvikantSrivastava/100-days-of-ML-Code/raw/master/Day12%20:%20Yolo/output.jpg)

**References**

- [YOLO: Real-Time Object Detection	](https://pjreddie.com/darknet/yolo/)
- [YOLO object detection with OpenCV - PyImageSearch	](https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/	)



## Day 13 : April 11, *Regularized linear Models*

**Today's Progress:** Learned about different ways to reduce overfitting in a model. Studied 3 different methods in depth. Read about the working of the following methods with mathematic understanding.
- Ridge Regression
- Lasso Regression
- Elastic Net
  

**Thoughts:** Overfitting can be avoided using the Regularized Models, having a good understanding is a icing on the cake.

**References**

- [Hands–On Machine Learning with Scikit–Learn and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)



## Day 14 : April 12, *Logistic Regression*

**Today's Progress:** Read about Logistic Regression. Logistic Regression (also called Logit Regression) is commonly used to estimate the probability that an instance belongs to a particular class.
- Estimating Probabilities
- Training and Cost Function
- Decision Boundaries

**References**

- [Hands–On Machine Learning with Scikit–Learn and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)



## Day 15 : April 13, *Binary Classifier Anime face vs Human face*

**Today's Progress:** Made a binary classifier on a custom dataset. Did a little web scraping to collect anime faces and human faces. 

**Thoughts:** Looking forward to work on GANs with the same dataset.

**Link of Work:**
  ![](https://raw.githubusercontent.com/AvikantSrivastava/100-days-of-ML-Code/master/Day15%20%3A%20Human%20vs%20Anime%20Face/download.png)
- [[python code]](https://github.com/AvikantSrivastava/100-days-of-ML-Code/blob/master/Day15%20:%20Human%20vs%20Anime%20Face/test1.py)
- [[ipynb]](https://github.com/AvikantSrivastava/100-days-of-ML-Code/blob/master/Day15%20:%20Human%20vs%20Anime%20Face/anime-vs-human.ipynb)


## Day 16 : April 14, *Support Vector Machines Introduction*
**Today's Progress:** Started off with reading about SVMs (Support Vector Machines) and got an intuition on what SVM is and how it is used to solve a Classification problem.

**References**
- [Hands–On Machine Learning with Scikit–Learn and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)


## Day 17 : April 15, *Softmargin SVM Classification*
**Today's Progress:** Implemented Softmargin classification on IRIS dataset in sklearn library.

**Link of Work:** [[python code]](https://github.com/AvikantSrivastava/100-days-of-ML-Code/blob/master/Day17%20:%20Softmargin%20SVM%20Classification/iris%20dataset%20using%20svm.md)

## Day 18 : April 16, *Backpropagation in Neural Networks*
**Today's Progress:** I tried to dig deeper and understood the mathematics behind the backpropagation algorithm. Learned about the calculus, vectorization and cost function behind every neuron in a network.

**Work**
![](https://github.com/AvikantSrivastava/100-days-of-ML-Code/raw/master/Day18%20:%20Backpropagation/backprop1.jpeg)

![](https://github.com/AvikantSrivastava/100-days-of-ML-Code/raw/master/Day18%20:%20Backpropagation/backprop2jpeg)

**References** 
- [How Does Back-Propagation in Artificial Neural Networks Work?](https://towardsdatascience.com/how-does-back-propagation-in-artificial-neural-networks-work-c7cad873ea7)
- [What is backpropagation really doing? Deep learning, chapter 3](https://www.youtube.com/watch?v=Ilg3gGewQ5U)

<!-- ## Day 19 : April 17, *Support Vector Machines* -->







<!-- 

template
## Day 0 : month day, *name*

**Today's Progress:** haha

**Thoughts:** haha

**Link of Work:**
- haha
**References**

- haha

 -->














<!-- ## Day 7 : April 5 , **
- https://arxiv.org/pdf/1506.02640.pdf
- https://medium.com/@enriqueav/object-detection-with-yolo-implementations-and-how-to-use-them-5da928356035
- https://cv-tricks.com/object-detection/faster-r-cnn-yolo-ssd/
 -->


<!-- ## Day # : ########## *Dog Breed*

**References**
- https://medium.com/@claymason313/dog-breed-image-classification-1ef7dc1b1967
- https://medium.com/@utsumuki_neko/using-inception-v3-from-tensorflow-hub-for-transfer-learning-a931ff884526
- https://medium.com/tensorflow/introducing-tensorflow-datasets-c7f01f7e19f3
- https://towardsdatascience.com/dog-breed-classification-hands-on-approach-b5e4f88c333e
- https://www.tensorflow.org/api_docs/python/tf/keras/applications/InceptionV3
- https://github.com/novasush/Parallelization-with-TFDS/blob/master/TFDS_Parallel.ipynb -->

<!-- tools to edit markdown -->
<!-- http://tools.buzzstream.com/meta-tag-extractor -->
