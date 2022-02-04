import numpy
import scipy.special
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk import word_tokenize
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from collections import Counter


stop_words_nltk = stopwords.words('english')


def read_csv_data(path):
    """
    Read data as dataframe and return needed column(s) as list(s)
    
    :param path: path to csv file with data
    :param column_text: name/header of column with the text
    :param column_label: if file has gold label, name/header of column with the gold label
    
    returns: text and label or text
    """
    data = pd.read_csv(path)
    text = list(data['text'])    
    labels = list(data['airline_sentiment'])
    print('There are ', len(text), 'tweets')
    print('Distribtion of the labels:', Counter(labels))
    return text, labels


class neuralNetwork:
    
    #initialization function to create network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        """
        Applies the parameters to the neural network object, network has one hidden layer.
        
        :param inputnodes: number of nodes in input layer
        :param hiddennodes: number of nodes in hidden layer
        :param outputnodes: number of nodes in output layer
        :param learningrate: learning rate of gradient descent        
        :type inputnodes: int
        :type hiddennodes: int
        :type outputnodes: int
        :type learningrate: int
        
        returns: None
        """

        self.inodes = inputnodes 
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        
        #matrix of random weigths (w) that link input layer (i) to hidden layer (h)
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        
        #matrix of random weights (w) that link hidden layer (h) to output later (o)
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
        
        # defining the activation function
        self.activation_function = lambda x: scipy.special.expit(x)  
        
        pass
    
    # training phase
    def train(self, inputs_list, targets_list):
        """
        Coverts input to output through the network and backpropagates the error to update weights, thereby training the NN
        
        :param inputs_list: training features
        :param targets_list: training labels or target output
        :type inputs_list: list or one dimensional array
        :type targets_list: list or one dimensional array
        
        returns: None
        """
        
        ###part 1: going from input to output:
        
        # convert inputs to 2d array 
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        # signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        ###part 2: backpropagating error
        
        # calculate errors in both layers
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors) 
        
        # update the weights for hidden-output
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        
        # update the weights for input-hidden
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        pass
    
    # query fase
    def query(self, inputs_list):
        """
        Gives predictions based on input. 
        
        inputs_list: list 
        returns: array with output
        """
        # convert inputs_list to matrix, and transposes so input is a column not a row
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        # calculate signals into hidden layer: use matrix multiplication of weights and input
        hidden_inputs = numpy.dot(self.wih, inputs)
        # apply activation (sigmoid) to calculate what comes out of hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer: matrix multiplication of weights and input (in this case output from previous layer)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # apply activation (sigmoid) to calculate what comes out of output layer
        final_outputs = self.activation_function(final_inputs)
        
        
        return final_outputs 

    
# function to turn labels into arrays  
def transform_labels(labels):
    """
    Transforms textual label into array.
    
    :param labels: gold labels
    :type labels: list
    
    :returns: targets (label in array form)
    """
    label_dict = {'negative': 0, 'neutral': 1, 'positive': 2}
    targets = []
    for label in labels:
        new_label = numpy.zeros(3) + 0.01
        new_label[label_dict[label]] = 0.99
        targets.append(new_label)
    return targets




## tfidf approach

# making training data into arrays
def transform_training_data(data):
    """
    Creates and fits TF-IDF vectorizer to training data and transforms training data
    
    :param data: training data
    :type data: list
    
    :returns: training_vec (transformed training data), tfidf (fitted)
    """

    tfidf = TfidfVectorizer(tokenizer = word_tokenize, stop_words=stop_words_nltk, min_df=10)
    training_vec_tfidf = tfidf.fit_transform(data)
    svd = TruncatedSVD(n_components=300)
    training_vec = svd.fit_transform(training_vec_tfidf)
    return training_vec, tfidf, svd

# making test data into arrays
def transform_test_data(data, tfidf, svd):
    """Transforms textual data into input for network.
    
    :param data: textual test data
    :param tfidf_vec: tf idf vectorizer fitted to training data
    :type data: list
    
    :returns: test_vec (array)
    """

    test_vec_tfidf = tfidf.transform(data)
    test_vec = svd.transform(test_vec_tfidf)
    
    return test_vec



## word embedding approach

# making data into arrays (same for training and test)
# code partially taken from intro to hlt course
def tweet_embedding(data, path_model):
    """
    Transforms textual data to embedding of sentences/tweets
    
    :param data:
    :param path_model: path to language model (bin)
    :type data: list
    :type path_model: str
    
    :returns: embedding data (list)
    """
    embedding_data = []
    
    word_embedding_model = KeyedVectors.load_word2vec_format(path_model, binary=True)
    
    for tweet in data:
        # array of right length with zeros
        tweet_emb = numpy.zeros(300,dtype="float32")
        #counter of number of words
        n_words = 0
        #tokenize
        words = word_tokenize(tweet)
        for word in words:
            if word not in stop_words_nltk:
                if word in word_embedding_model:
                    tweet_emb = numpy.add(tweet_emb, word_embedding_model[word])
                    n_words += 1
                else:
                    word = word.lower()
                    if word in word_embedding_model:
                        tweet_emb = numpy.add(tweet_emb, word_embedding_model[word])
                        n_words += 1
        
        # get average of vector using counter
        if n_words > 0:
            tweet_emb = numpy.divide(tweet_emb, n_words)

        embedding_data.append(tweet_emb)
        
    return embedding_data

def prepare_data(training_file, test_file, mode, language_model):
    
    #training data
    print('Loading training data...')
    training_text, training_labels = read_csv_data(training_file)
    if mode == 'tfidf':
        training_data, tfidf, svd = transform_training_data(training_text)
    if mode == 'embeddings':
        training_data = tweet_embedding(training_text, language_model)
    
    training_targets = transform_labels(training_labels)

    # test data
    print('Loading test data...')
    test_text, test_labels = read_csv_data(test_file)
    #transform training data
    if mode == 'tfidf':
        test_data = transform_test_data(test_text, tfidf, svd)
    if mode == 'embeddings':
        test_data = tweet_embedding(test_text, language_model)

    #test_targets = transform_labels(test_labels)

    return training_data, training_targets, test_data, test_labels


def train_test(training_data, training_targets, test_data, test_labels, input_n, hidden_n, output_n, lr, epochs):

    # nn
    print('Making network...')
    nn = neuralNetwork(input_n, hidden_n, output_n, lr)

    # train nn
    print('Training network...')
    for e in range(epochs): 
        for tweet, label in zip(training_data, training_targets): 
            nn.train(tweet, label)

    # test nn
    print('Querying network with test data...')
    all_predictions = []

    label_dict_inverse = {0: 'negative', 1:'neutral', 2:'positive'}

    for tweet in test_data:
        outputs = nn.query(tweet)
    
        # the index of the highest value corresponds to the label
        pred_index = numpy.argmax(outputs)
        pred = label_dict_inverse[pred_index]
        all_predictions.append(pred)

    #have full classification scores as return so it can be looked at later if needed
    cr = classification_report(test_labels, all_predictions, digits=4)
    #only print the f1 for now as this is the only score needed for most of the models
    print(f1_score(test_labels, all_predictions, average='macro'))
    
    return cr
