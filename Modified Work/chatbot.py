import nltk
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import tokenizer_from_json
import numpy as np

from keras.models import load_model
import json
import random
from tkinter import *

model = load_model('chatbot_model.h5')


intents = json.loads(open('intents.json').read())
context=""
context1=""
padding_type = 'post'
trunc_type = 'post'

with open('tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)
with open('tokenizer_label.json') as f:
    data = json.load(f)
    label_tokenizer = tokenizer_from_json(data)


data_file = open('intents.json').read()
intents = json.loads(data_file)
Questions = []
labels=['greeting', 'goodbye', 'thanks', 'options', 'adverse_drug', 'blood_pressure', 'blood_pressure_search', 'pharmacy_search', 'hospital_search']
Answers = []


def predict_class(sentence):
    # filter out predictions below a threshold
    sentence=[sentence]
    p = tokenizer.texts_to_sequences(sentence)
    padded = pad_sequences(p, maxlen=10, padding=padding_type, truncating=trunc_type)
    res = model.predict(padded)
    result2=np.argmax(res)
    result=labels[result2-1]
    return result


def getResponse(ints, intents_json):
    tag = ints
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if (i['tag'] == tag):
            result = random.choice(i['responses'])
            global context1
            context1= i['context'][0]
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg)
    print("Context "+ints)
    if (context1!= ""):
        ints = context1
    res = getResponse(ints.strip(), intents)
    return res


# Creating GUI with tkinter



def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)
    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12))
        global context
        if(context=="goodbye"):
            with open("Chatlog.txt", "a+") as file_object:
                # Move read cursor to the start of file.
                file_object.seek(0)
                # If file is not empty then append '\n'
                data = file_object.read(100)
                if len(data) > 0:
                    file_object.write("\n")
                # Append text at the end of file
                file_object.write("User: " + msg + "\n")
                file_object.write("\n\n\n")
                ChatLog.insert(END, "Bot: " + "Thank you for your rating" + '\n\n')
            context = ""

        else:
            res = chatbot_response(msg)
            ChatLog.insert(END, "Bot: " + res + '\n\n')
            with open("Chatlog.txt", "a+") as file_object:
                # Move read cursor to the start of file.
                file_object.seek(0)
                # If file is not empty then append '\n'
                data = file_object.read(100)
                if len(data) > 0:
                    file_object.write("\n")
                # Append text at the end of file
                file_object.write("User: " + msg + "\n")
                file_object.write("Bot: " + res)
                context = predict_class(msg)
                if (context == "goodbye"):
                    if (context1 == ""):
                        ChatLog.insert(END, "Bot: Please provide a rating for this chat out of 5\n\n")
                        file_object.write("\nBot: Please provide a rating for this chat out of 5")
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


base = Tk()
base.title("Hello")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

# Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial", )

ChatLog.config(state=DISABLED)

# Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

# Create Button to send message
SendButton = Button(base, font=("Verdana", 12, 'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b", fg='#ffffff',
                    command=send)

# Create the box to enter message
EntryBox = Text(base, bd=0, bg="white", width="29", height="5", font="Arial")
# EntryBox.bind("<Return>", send)


# Place all components on the screen
scrollbar.place(x=376, y=6, height=386)
ChatLog.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=6, y=401, height=90, width=265)
SendButton.place(x=281, y=401, height=90, width=110)

base.mainloop()
