import io

import nltk
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
import json
import re
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, Embedding, Bidirectional


words=[]
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)
#oov_tok = '<OOV>'
padding_type = 'post'
trunc_type = 'post'
embedding_dim = 256
vocab_size = 256
classes = []


Questions = []

Answers = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        Answers.append(intent['tag'])
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
        Questions.append(pattern.lower())

print(len(Answers))
print(len(Questions))


train_size = int(len(Questions) * 1.0)

train_questions = Questions[0: train_size]
train_tags = Answers[0: train_size]
print(len(train_questions))
print(len(train_tags))


tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_questions)
word_index = tokenizer.word_index

print(dict(list(word_index.items())))

train_sequences = tokenizer.texts_to_sequences(train_questions)

print(train_sequences[10])
train_padded = pad_sequences(train_sequences, maxlen=10, padding=padding_type, truncating=trunc_type)


print(set(Answers))
label_tokenizer = Tokenizer(filters="")
label_tokenizer.fit_on_texts(Answers)
print(train_tags)
training_label_seq = np.array((label_tokenizer.texts_to_sequences(train_tags)))
print(label_tokenizer.index_word)
print(training_label_seq)
model = Sequential([
    # Add an Embedding layer expecting input vocab of size 256, and output embedding dimension of size 256 we set at the top
    Embedding(vocab_size, embedding_dim),
    Bidirectional(LSTM(embedding_dim)),
    # use ReLU in place of tanh function since they are very good alternatives of each other.
    Dense(embedding_dim, activation='relu'),
    # Add a Dense layer with 10 units and softmax activation.
    # When we have multiple outputs, softmax convert outputs layers into a probability distribution.
    Dense(10, activation='softmax')
])
model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
num_epochs = 100
train_padded= np.asarray(train_padded).astype(np.float32)
training_label_seq= np.asarray(training_label_seq).astype(np.float32)

print(train_padded)
history = model.fit(train_padded, np.array(training_label_seq), batch_size=5, epochs=num_epochs, verbose=1)
model.save('chatbot_model.h5', history)
print("model created")


tokenizer_json = tokenizer.to_json()
with io.open('tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))

label_tokenizer_json = label_tokenizer.to_json()
with io.open('tokenizer_label.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(label_tokenizer_json, ensure_ascii=False))

print("tokenizer saved")

labels=['greeting', 'goodbye', 'thanks', 'options', 'adverse_drug', 'blood_pressure', 'blood_pressure_search', 'pharmacy_search', 'hospital_search']

txt = ["Hi"]
seq = tokenizer.texts_to_sequences(txt)
padded = pad_sequences(seq, maxlen=10, padding=padding_type, truncating=trunc_type)
pred = model.predict(padded)
print(padded)

print("Hi \n")
print(labels[np.argmax(pred)-1])

txt = ["What can you do for me?"]
seq = tokenizer.texts_to_sequences(txt)
padded = pad_sequences(seq, maxlen=10, padding=padding_type, truncating=trunc_type)
pred = model.predict(padded)
print(padded)

print("What can you do for me? \n")
print(labels[np.argmax(pred)-1])

txt = ["Search blood pressure patient"]
seq = tokenizer.texts_to_sequences(txt)
padded = pad_sequences(seq, maxlen=10, padding=padding_type, truncating=trunc_type)
pred = model.predict(padded)
print(padded)

print("Search blood pressure patient \n")
print(labels[np.argmax(pred)-1])

txt = ["search hospital data"]
seq = tokenizer.texts_to_sequences(txt)
padded = pad_sequences(seq, maxlen=10, padding=padding_type, truncating=trunc_type)

print(padded)

pred = model.predict(padded)
print("Search hospital \n")
print(labels[np.argmax(pred)-1])

txt = ["Search Pharmacy"]
seq = tokenizer.texts_to_sequences(txt)
padded = pad_sequences(seq, maxlen=10, padding=padding_type, truncating=trunc_type)
pred = model.predict(padded)
print(padded)

print("Search Pharmacy \n")
print(labels[np.argmax(pred)-1])

txt = ["Check adverse Drug"]
seq = tokenizer.texts_to_sequences(txt)
padded = pad_sequences(seq, maxlen=10, padding=padding_type, truncating=trunc_type)
pred = model.predict(padded)
print(padded)

print("Check Adverse Drug \n")
print(labels[np.argmax(pred)-1])
