#!/usr/bin/env python3

"""
Ziyi Wang z5248041

Brief answer:

In general, the preprocessing is replacing contractions and symbols with complete words, and the network is two Bi-LSTM
with attention with a linear output layer to predict rating and category separately, and the loss function is
CrossEntropy, and the optimiser is Adam.

For tokenise and postprocessing parts, There are not additional methods needed to be implemented. The stopwords and
punctuations are not removed as the attention mechanism is implemented and removing stopwords and punctuations does not
improve the weighted score in this case. The dimension of GloVe is changed to 200 as it has good performances in the
experiments.

For the network, I choose bidirectional LSTM as it can learn the words relationship in sentences. Futhermore, combined
by with the attention mechanism, it has better performance than GRU or CNN models on this dataset. After some tests, I
found that using two LSTM models to learn rating and category separately tends to get higher score. The reason may be
that the model can learn different weights for each word based on different tasks. Both LSTM have 200 of hidden size and
2 layers, but the dropout is 0.5 for rating LSTM and 0.4 for category LSTM as I think more information are needed to
pass on to get a better classification result for the category task. The attention is implemented by a linear function
followed by tanh the multiply by u_omega followed by softmax then multiply by output of LSTM then do the sum. This can
help the model to learn the attention in a sentence and improve the performance. There is a dropout of 0.3 for the input
data of the model to prevent the over fitting.

For the loss function, the corssentropy is chosen as it is good for classification tasks. After experiments, I notice
that the accuracy of category is lower than that of rating. Hence a hyper parameter sigma is set at 0.6 to lower the
weight of category loss, this can improve the performance slightly.For the convert part, I use the argmax to find the
max value to be the final prediction. This combination is simple but efficient. For the training part, batchsize is 128
and epochs is 20, and the optimiser is Adam with lr 0f 0.003. This setting can converge quickly and avoid local minima.
"""

"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables stopWords,
wordVectors, trainValSplit, batchSize, epochs, and optimiser, as well as a basic
tokenise function.  You are encouraged to modify these to improve the
performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.
You may change this variable in the config.py file.

You may only use GloVe 6B word vectors as found in the torchtext package.
"""

import torch
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe
import re
# import numpy as np
# import sklearn

from config import device

################################################################################
##### The following determines the processing of input data (review text) ######
################################################################################

def tokenise(sample):
    """
    Called before any processing of the text has occurred.
    """

    processed = sample.split()
    return processed

def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """

    content = ' '.join(sample)
    content = re.sub(r"\'ll", " will", content)
    content = re.sub(r"\'d", " would", content)
    content = re.sub(r"\'s", " is", content)
    content = re.sub(r"\'m", " am", content)
    content = re.sub(r"\'ve", " have", content)
    content = re.sub(r"\'re", " are", content)
    content = content.replace('&', 'and')
    content = content.replace('$', '')
    content = content.split()
    return content

def postprocessing(batch, vocab):
    """
    Called after numericalising but before vectorising.
    """

    return batch

stopWords = {}
wordVectors = GloVe(name='6B', dim=200)

################################################################################
####### The following determines the processing of label data (ratings) ########
################################################################################

def convertNetOutput(ratingOutput, categoryOutput):
    """
    Your model will be assessed on the predictions it makes, which must be in
    the same format as the dataset ratings and business categories.  The
    predictions must be of type LongTensor, taking the values 0 or 1 for the
    rating, and 0, 1, 2, 3, or 4 for the business category.  If your network
    outputs a different representation convert the output here.
    """

    ratingOutput = torch.argmax(ratingOutput, dim=-1)
    categoryOutput = torch.argmax(categoryOutput, dim=-1)
    return ratingOutput, categoryOutput

################################################################################
###################### The following determines the model ######################
################################################################################

class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network will be a
    batch of reviews (in word vector form).  As reviews will have different
    numbers of words in them, padding has been added to the end of the reviews
    so we can form a batch of reviews of equal length.  Your forward method
    should return an output for both the rating and the business category.
    """

    def __init__(self):
        super(network, self).__init__()
        self.dropout = tnn.Dropout(p=0.4)
        self.lstm_rat = tnn.LSTM(input_size=200, hidden_size=200, num_layers=2, batch_first=True,
                                 dropout=0.5, bidirectional=True)
        self.lstm_cat = tnn.LSTM(input_size=200, hidden_size=200, num_layers=2, batch_first=True,
                                 dropout=0.4, bidirectional=True)
        # self.w_omega = tnn.Parameter(torch.Tensor(2 * 200, 2 * 200))
        # self.b_omega = tnn.Parameter(torch.Tensor(2 * 100, 2 * 100))
        self.u_omega_rat = tnn.Parameter(torch.Tensor(100, 1))
        self.u_omega_cat = tnn.Parameter(torch.Tensor(100, 1))
        self.att_lin_rat = tnn.Linear(2 * 200, 100)
        self.att_lin_cat = tnn.Linear(2 * 200, 100)
        self.lin_rat = tnn.Linear(2 * 200, 2)
        self.lin_cat = tnn.Linear(2 * 200, 5)

        # tnn.init.uniform_(self.w_omega, -0.1, 0.1)
        # tnn.init.uniform_(self.b_omega, -0.1, 0.1)
        tnn.init.uniform_(self.u_omega_rat, -0.1, 0.1)
        tnn.init.uniform_(self.u_omega_cat, -0.1, 0.1)

    def att_rat(self, h):

        u = torch.tanh(self.att_lin_rat(h))
        alfa = tnn.functional.softmax(torch.matmul(u, self.u_omega_rat), dim=1)
        s = torch.sum(alfa * h, dim=1)
        return s

    def att_cat(self, h):

        u = torch.tanh(self.att_lin_cat(h))
        alfa = tnn.functional.softmax(torch.matmul(u, self.u_omega_cat), dim=1)
        s = torch.sum(alfa * h, dim=1)
        return s

    def forward(self, input, length):
        input = self.dropout(input)
        lstm_ro, (c_0, h_0) = self.lstm_rat(input)
        lstm_co, (c_0, h_0) = self.lstm_cat(input)
        # lstm_ro = torch.cat((lstm_ro[:, 0, :], lstm_ro[:, -1, :]), dim=-1).unsqueeze(1)
        # lstm_co = torch.cat((lstm_co[:, 0, :], lstm_co[:, -1, :]), dim=-1).unsqueeze(1)
        att_rat = self.att_rat(lstm_ro)
        att_cat = self.att_cat(lstm_co)
        lin_rat = self.lin_rat(att_rat.view(-1, 400))
        lin_cat = self.lin_cat(att_cat.view(-1, 400))
        return lin_rat, lin_cat


class loss(tnn.Module):
    """
    Class for creating the loss function.  The labels and outputs from your
    network will be passed to the forward method during training.
    """

    def __init__(self):
        super(loss, self).__init__()
        self.cel = tnn.CrossEntropyLoss()

    def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):
        loss_rat = self.cel(ratingOutput, ratingTarget)
        loss_cat = self.cel(categoryOutput, categoryTarget)
        return loss_rat + sigma * loss_cat


sigma = 0.6

net = network()
lossFunc = loss()

################################################################################
################## The following determines training options ###################
################################################################################

trainValSplit = 0.8
batchSize = 128
epochs = 15
# optimiser = toptim.SGD(net.parameters(), lr=0.01)
optimiser = toptim.Adam(net.parameters(), lr=0.003)
