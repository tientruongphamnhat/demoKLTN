from flask import Flask, render_template, request, jsonify
#import tensorflow as tf
from tensorflow import keras
import time
import io
import os
import numpy as np
import re
import unicodedata
from sklearn.model_selection import train_test_split
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import tensorflow as tf
from flask_cors import CORS

app = Flask('translate')
CORS(app)
# -----


# tf.enable_eager_execution()


sess = tf.compat.v1.Session()
graph = tf.compat.v1.get_default_graph()


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    #w = unicode_to_ascii(w.lower().strip())
    w = w.lower()
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    #w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w = w.strip()
    w = '<start> ' + w + ' <end>'
    return w


# load vocab
with open('en_vocab.txt', 'r', encoding="utf8") as f:
    word_en = f.read().split('\n')

with open('vi_vocab.txt', 'r', encoding="utf8") as f:
    word_vi = f.read().split('\n')

word2int_en = dict()
word2int_vi = dict()
int2word_vi = dict()

j = 1
for i in word_en:
    word2int_en[i] = j
    j += 1

k = 1
for i in word_vi:
    word2int_vi[i] = k
    k += 1

l = 1
for i in word_vi:
    int2word_vi[l] = i
    l += 1

# Hyperpamaters
BATCH_SIZE = 64
embedding_dim = 256
units = 128
vocab_inp_size = len(word_en) + 1
vocab_tar_size = len(word_vi) + 1

max_length_inp = 169
max_length_targ = 210
# Model
# Encoder


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

# Attention


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


attention_layer = BahdanauAttention(10)
# Decoder


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights


decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

optimizer = tf.keras.optimizers.Adam()
# Checkpoints
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

# translate


def evaluate(sentence):
    attention_plot = np.zeros((max_length_targ, max_length_inp))

    sentence = preprocess_sentence(sentence)

    # check word
    words = sentence.split(" ")
    for word in words:
        if word not in word_en:
            sentence = sentence.replace(word, "<unk>")
    #

    inputs = [word2int_en[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=max_length_inp,
                                                           padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([word2int_vi['<start>']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)

        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += int2word_vi[predicted_id] + ' '

        if int2word_vi[predicted_id] == '<end>':
            return result, sentence, attention_plot

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot


# with sess.as_default():
#    with graph.as_default():

def predict(sentence):
    result, sentence, attention_plot = evaluate(sentence)
    return result


# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


@app.route('/')
def show_predict_stock_form():
    # return render_template('predictorform.html')
    # return render_template('try.html')
    return "hello"


@app.route('/translate', methods=['POST'])
def translate():
    if request.method == 'POST':
        if request.json.get('input') == '':
            return jsonify({'message': 'input is null'}), 400
        else:
            input = (request.json.get('input')).strip()

            inputArray = []
            index = 0
            temp = 0
            lenInput = len(input)
            output = ''
            while index < lenInput:
                if(input[index] == '.' or input[index] == '?' or input[index] == '!'):
                    strTemp = input[temp: index + 1]
                    strTemp = strTemp.strip()
                    inputArray.append(strTemp)
                    temp = index + 1
                    index += 1
                    continue
                if(index == lenInput-1):
                    strTemp = input[temp: index + 1]
                    strTemp = strTemp.strip()
                    inputArray.append(strTemp)
                index += 1

            print(inputArray)
            for i in inputArray:
                output += (predict(i).replace('<end>', ""))

            output.replace('  ', ' ')
            output.replace(' ?', '?')
            output.replace(' .', '.')
            # print(output)
        return jsonify({'output': output}), 200


app.run("localhost", "9999", debug=True)
