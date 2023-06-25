import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy
import tflearn
import tensorflow
import random
import json
import pickle
from tensorflow import keras
import time
import datetime
import os



# Loading our intents for processing.
with open("expected_intent.json") as file:
    data = json.load(file)
print(data['intents'])
print(len(data['intents']))

print("Press 1. to Retrain the model on Updated intents")
print("Press 2. to start Chatting with the bot")
val = input(" Enter a value :- ")

def createModel():
    words = []
    labels = []
    traning = []
    output = []
    with open("data.pickle","rb") as f:
        words, labels, traning, output = pickle.load(f)
    net = tflearn.input_data(shape=[None, len(traning[0])]) #define the input shape each training length will be same
    net = tflearn.fully_connected(net, 8) #8 neurons in the hidden layer
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output[0]), activation="softmax") #output layer allow us to get probabilities for each network
    net =  tflearn.regression(net) #regression is used to get probabilities
    model = tflearn.DNN(net)
    model.load("model.tflearn")
    return model
    
def trainingModel():
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]: #stemming words
            wrds = nltk.word_tokenize(pattern) #tokenizing words
            words.extend(wrds) #extending the list
            docs_x.append(wrds)
            docs_y.append(intent["tag"]) 

        if intent["tag"] not in labels:
            labels.append(intent["tag"])
    #how many words it has seen already 
    #vocab making
    words = [stemmer.stem(w.lower()) for w in words if w != "?"] # removing "?"
    words = sorted(list(set(words))) #make sure no duplicates

    labels = sorted(labels)

    #traning and testing data   
    #we have string and neural network understand only numbers
    #so we need to convert words to numbers
    #we create a bag of words
    #we are using one hot encoding

    traning = []
    output = []
    out_empty = [0 for _ in range(len(labels))]

    for x,doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                #word eixst in the patterns
                bag.append(1)
            else:
                bag.append(0)
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        traning.append(bag)
        output.append(output_row)

    #take these list and convert them to numpy arrays
    traning = numpy.array(traning)
    output = numpy.array(output)

    with open("data.pickle","wb") as f:
        pickle.dump((words, labels, traning, output), f)

    net = tflearn.input_data(shape=[None, len(traning[0])]) #define the input shape each training length will be same
    net = tflearn.fully_connected(net, 8) #8 neurons in the hidden layer
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output[0]), activation="softmax") #output layer allow us to get probabilities for each network
    net =  tflearn.regression(net) #regression is used to get probabilities

    model = tflearn.DNN(net)
    model.fit(traning, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")
    chat(model)
#Prediction from the user
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s) #list of tokenize words
    s_words = [stemmer.stem(word.lower()) for word in s_words] #stemming words
    
    #take each word and check if it is in the words list
    for se in s_words:
        for i,w in enumerate(words):
            if w == se:
                bag[i] = 1
    return numpy.array(bag)

#if we want to chat with the user
def chat(model):
    model.load("model.tflearn")
    with open("data.pickle","rb") as f:
        words, labels, traning, output = pickle.load(f)
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break     

        results = model.predict([bag_of_words(inp, words)])[0]
        
        # print(results) #this is the probability of each tag
        #pick the greatest from here
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        if results[results_index] > 0.7:
            # print(tag)
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
            print(random.choice(responses))
        else:
            print("I didn't get that, try again.")

if val == "1":
    trainingModel()
    
elif val == "2":
    model = createModel()
    chat(model)




