# Sentiment Analysis on Disaster Tweets

<img src="/readme/twitter.webp" align="left" width="220px" height="192px"/>
<img align="left" width="0" height="192px" hspace="10"/>

> This project is created for detecting disaster tweets using natural language processing.

> The project aims to predict the sentiment (whether a tweet is referring to a disaster or not) using Natural Language Processing (NLP) techniques and deep learning models. The dataset used for this project is taken from the [Kaggle competition](https://www.kaggle.com/c/nlp-getting-started/data).
<br>
<br>


## Project Overview

The project is divided into the following major steps:

1. Importing the dataset: The dataset is imported using the pandas library and only the 'text' and 'target' columns are used.

2. Data Preprocessing: This step involves text preprocessing using regular expressions to remove unnecessary characters and stopwords to remove common words that do not carry much meaning. In addition, lemmatization is used to group together the inflected forms of a word so they can be analyzed as a single item.

3. Splitting the dataset: The dataset is split into a training set and a validation set to train and evaluate the model.

4. Embedding Hyperparameters: This step sets up the hyperparameters used for the embedding layer in the deep learning model, such as the maximum vocabulary size, embedding dimension, and padding/truncating length.

5. Tokenization and Padding: This step involves tokenizing the text data and padding them to make them uniform in length.

6. Model Creation: A deep learning model is created using the Keras library, with an embedding layer, bidirectional LSTM layer, dense layers, and a final output layer with a sigmoid activation function.

7. Model Training: The model is trained on the training dataset with the Adam optimizer and binary cross-entropy loss function. The training process is monitored using early stopping and model checkpoint callbacks to prevent overfitting and save the best model.

8. Model Evaluation: The performance of the model is evaluated using accuracy and loss metrics. The results are visualized using graphs to determine the effectiveness of the model.


## Requirements
The following libraries are required to run this project:

* Tensorflow
* Keras
* NLTK
* Pandas
* Numpy
* Matplotlib
* Spacy
* tqdm
* Wordcloud


## Installing the Required Libraries

```sh
$ pip3 install requirements.txt
```

## Usage
To use this project, simply run the provided code in a Python environment that has the required libraries installed.

To test your own tweets, you should create a test.csv file. Then under "text" column, you can write as many tweets as you want and run the cells after training the model.


## License
This project is licensed under the [Apache-2.0 License](/LICENSE).
