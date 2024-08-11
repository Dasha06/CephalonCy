import tensorflow as tf
import numpy as np
import tflearn
import nltk
import json
import pickle
nltk.download('punkt_tab')


with open('CephalonCyQuotes.json') as file:
    data = json.load(file)


# Tokenize the text
words = []
for dialogue in data:
    sentence_words = nltk.word_tokenize(dialogue)
    words.extend(sentence_words)
# Remove any punctuation and special characters
words = [word for word in words if word.isalnum()]
# Convert all the text to lowercase
words = [word.lower() for word in words]

print(words)

# Create a dictionary of words and their frequencies
word_freq = nltk.FreqDist(words)

# Get the most common words
common_words = word_freq.most_common(1000)
# Create a list of the most common words
word_list = [word[0] for word in common_words]
# Create a dictionary of words and their index in the word list
word_dict = {word: index for index, word in enumerate(word_list)}
# Create the training data
training_data = []
for dialogue in data:
    for i in range(len(dialogue) - 1):
        input_sentence = dialogue[i]
        output_sentence = dialogue[i + 1]

        # Tokenize the input and output sentences
        input_words = nltk.word_tokenize(input_sentence)
        output_words = nltk.word_tokenize(output_sentence)

        # Remove any punctuation and special characters
        input_words = [word for word in input_words if word.isalnum()]
        output_words = [word for word in output_words if word.isalnum()]

        # Convert the input and output sentences to vectors of numbers
        input_vector = [0] * len(word_list)
        for word in input_words:
            if word in word_dict:
                index = word_dict[word]
                input_vector[index] = 1

        output_vector = [0] * len(word_list)
        for word in output_words:
            if word in word_dict:
                index = word_dict[word]
                output_vector[index] = 1

        # Add the input and output vectors to the training data
        training_data.append([input_vector, output_vector])


# Build the neural network model
net = tflearn.input_data(shape=[None, len(word_list)])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(word_list), activation='softmax')
net = tflearn.regression(net)

# Create the model
model = tflearn.DNN(net)

# Train the model
model.fit([data[0] for data in training_data], [data[1] for data in training_data], n_epoch=1000, batch_size=8, show_metric=True)

# save the model
filename = 'linear_model.sav'
pickle.dump(model, open(filename, 'wb'))

# load the model
load_model = pickle.load(open(filename, 'rb'))

def get_response(question):
    # Tokenize the input question
    question_words = nltk.word_tokenize(question)

    # Remove any punctuation and special characters
    question_words = [word for word in question_words if word.isalnum()]

    # Convert the question to a vector of numbers
    question_vector = [0] * len(word_list)
    for word in question_words:
        if word in word_dict:
            index = word_dict[word]
            question_vector[index] = 1

    # Use the model to predict the response
    prediction = model.predict([question_vector])[0]
    response_vector = np.zeros(len(word_list))
    response_vector[np.argmax(prediction)] = 1

    # Convert the response vector to text
    response_words = []
    for index, value in enumerate(response_vector):
        if value == 1:
            response_words.append(word_list[index])

    response = ' '.join(response_words)
    return response

while True:
    question = input("You: ")
    response = get_response(question)
    print("AI: " + response)
