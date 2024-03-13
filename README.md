# EDA: Twitter Sentiment Analysis Using NN

## Table of Contents

- [Introduction](#introduction)
- [Environment Setup](#environment-setup)
- [Installation Instructions](#installation-instructions)
- [Resources](#resources)
- [License](#license)
- [Conclusion](#conclusion)
- [About Author](#about-author)

## Introduction

Computer vision methods and strategies can help to recognize the fruits with some basic features like the color of fruits, intensity of fruits ,shape of fruits and texture of the fruits. The term "recognize" is to predict the name of the fruit. In this project, we are going to use 81 different fruits claases. We will train the model using tensorflow.

### Problem Description/Definition

To build a robust system to recognize the fruits according to the color of fruits, intensity of fruits ,shape of fruits and texture of the fruits.

### Evaluation Measures

After training the model, we will apply the evaluation measures to check that how the model is getting predictions. We will use the following evaluation measures to evaluate the performance of the model:
    <li>Accuracy</li>
    <li>Plots of training and validation scores</li>

### Technical Approach

We are using python language in the implementations and Jupyter Notebook that support the machine learning and data science projects. We will build tensorflow based model. We will use Fruits360 dataset. The dataset providers provide the training and test data separately. After training on the model, we will evaluate the model to check the performance of trained model.

### Implementing Tensorflow Based Model for Training

<h4> Step 1</h4>
- We are calling base Sequancial model for training and for further tuning of parameters on image data. We must call it when we work on the keras, tensorflow based libraries.

<h4> Step 2</h4>
- Conv2D is 2D convolutional layer(where filters are applied to original image with specific features map to reduce the number of features), Conv2D layer create the convolution kernel(fixed size of boxes to apply on the image like below in the example gif) that take input of 16 filters which help to produce a tensor of outputs. We are giving input of the image with size of 100 width and 100 height and 3 is the channel for RGB.
<img src="https://miro.medium.com/max/1320/1*DTTpGlhwkctlv9CYannVsw.gif">


<h4> Step 3</h4>
- Activation function is node that is put at the end of all layers of neural network model or in between neural network layers. Activation function help to decide which neuron should be pass and which neuron should fire. So activation function of node defines the output of that node given an input or set of inputs. 
<img src="https://missinglink.ai/wp-content/uploads/2018/11/activationfunction-1.png">

<h4> Step 4</h4>
- Maxpooling is a pooling operation that calculates maximum value in each patch of each feature map. It takes the value from input vectors and prepare the vector for the next layers.
<img src="https://developers.google.com/machine-learning/practica/image-classification/images/maxpool_animation.gif">

<h4> Step 5</h4>
- Droupout layer drop some neurons from previous layers. why we apply this? We apply this to avoid the overfitting problems. In overfitting, model give good accuracy on training time but not good on testing time.
<img src="https://drek4537l1klr.cloudfront.net/elgendy/v-3/Figures/Img_01-04A_171.gif">

<h4> Step 6</h4>
- Flatten layer convert the 2D array into 1D array of all features.
<img src="https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/73_blog_image_1.png">

<h4> Step 7</h4>
- Dense layer reduce the outputs by getting inputs from Faltten layer. Dense layer use all the inputs of previous layer neurons and perform calculations and send 150 outputs

## Environment Setup

**Prerequisites**: Ensure Python 3.6 or newer is installed on your system.

1. **Create a Virtual Environment**:
    - Install `virtualenv` if you prefer it over the built-in `venv` (optional):
        ```bash
        pip install virtualenv
        ```
    - Create the environment:
        - With `venv` (Python 3.3+):
            ```bash
            python -m venv env
            ```
        - Or, with `virtualenv`:
            ```bash
            virtualenv env
            ```
    - Activate the environment:
        - Windows: `env\Scripts\activate`
        - Unix/MacOS: `source env/bin/activate`
    - To deactivate: `deactivate`

2. **Dependencies**:
    Ensure all dependencies are listed in `requirements.txt`. Install them using:
    ```bash
    pip install -r requirements.txt
    ```

## Installation Instructions

To use this project, clone the repository and set up the environment as follows:

1. **Clone the Repository**:
    ```bash
    https://github.com/Imran-ml/EDA-Twitter-Sentiment-Analysis-Using-NN.git
    ```
2. **Setup the Environment**:
    - Navigate to the project directory and activate the virtual environment.
    - Install the dependencies from `requirements.txt`.

## Resources

- **Kaggle Notebook**: [View Notebook](https://www.kaggle.com/code/muhammadimran112233/eda-twitter-sentiment-analysis-using-nn)
- **Dataset**: [View Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)

## License

This project is made available under the MIT License.

## Conclusion

We used the twitter sentiment analysis dataset and explored the data with different ways:
        <li>We prepared the text data of tweets by removing the unnecessary things.</li>
        <li>We trained model based on tensorflow with all settings. </li>
        <li>We evaluated thye model with different evaluation measures.</li>
        <li>If you are interested to work on any text based project, you can simply apply the same methodolgy but might be you will need to change little settings like name                 of coloumns etc.</li>
        <li>We worked on the classification problem and sepcifically we call it binary classification which is two class classification.</li>

## About Author

- **Name**: Muhammad Imran Zaman
- **Email**: [imranzaman.ml@gmail.com](mailto:imranzaman.ml@gmail.com)
- **Professional Links**:
    - Kaggle: [Profile](https://www.kaggle.com/muhammadimran112233)
    - LinkedIn: [Profile](linkedin.com/in/muhammad-imran-zaman)
    - Google Scholar: [Profile](https://scholar.google.com/citations?user=ulVFpy8AAAAJ&hl=en)
    - YouTube: [Channel](https://www.youtube.com/@consolioo)
- **Project Repository**: [GitHub Repo](https://github.com/Imran-ml/EDA-Twitter-Sentiment-Analysis-Using-NN.git)
