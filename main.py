import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.datasets import imdb
from keras_preprocessing.sequence import pad_sequences
from keras import models, layers
from keras import backend as K

class DataSet :
    def __init__(self) :
        vocab_size = 10000
        (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

        max_len = 500
        X_train = pad_sequences(X_train, maxlen=max_len)
        X_test = pad_sequences(X_test, maxlen=max_len)

        y_train = np.array(y_train)
        y_test = np.array(y_test)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

# class BahdanauAttention(models.Model) :
#     def __int__(self, units) :
#         super(BahdanauAttention, self).__init__()
#         self.W1 = layers.Dense(units)
#         self.W2 = layers.Dense(units)
#         self.V = layers.Dense(1)
#
#     def call(self, values, query) :
#         # query shape == (batch_size, hidden size)
#         # hidden_with_time_axis shape == (batch_size, 1, hidden size)
#         # score 계산을 위해 뒤에서 할 덧셈을 위해서 차원을 변경해줍니다.
#         hidden_with_time_axis = tf.expand_dims(query, 1)
#
#         # score shape == (batch_size, max_length, 1)
#         # we get 1 at the last axis because we are applying score to self.V
#         # the shape of the tensor before applying self.V is (batch_size, max_length, units)
#         score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
#
#         # attention_weights shape == (batch_size, max_length, 1)
#         attention_weights = tf.nn.softmax(score, axis = 1)
#
#         # context_vector shape after sum == (batch_size, hidden_size)
#         context_vector = attention_weights * values
#         context_vector = tf.reduce_sum(context_vector, axis=1)
#
#         return context_vector, attention_weights

class Attention(layers.Layer) :
    def __init__(self, return_squences = True) :
        self.return_sequences = return_squences
        super(Attention, self).__init__()

    def build(self, input_shape) :
        self.W = self.add_weight(name = 'att_weight', shape = (input_shape[-1], 1), initializer = 'normal')
        self.b = self.add_weight(name = 'att_bias', shape = (input_shape[1], 1), initializer = 'zeros')

        super(Attention, self).build(input_shape)

    def call(self, x) :
        e = K.tanh(K.dot(x, self.W) + self.b)

        a = K.softmax(e, axis = 1)
        output = x * a

        if self.return_sequences :
            return output

        return K.sum(output, axis = 1)


class Model(models.Model) :
    def __init__(self, max_len, vocab_size) :
        input = layers.Input(shape = (max_len, ), dtype = np.int32)
        embedding = layers.Embedding(vocab_size, 128, input_length = max_len, mask_zero = True)(input)

        lstm = layers.Bidirectional(layers.LSTM(64, dropout = 0.5, return_sequences = True))(embedding)
        # lstm, forward_h, forward_c, backward_h, backward_c = layers.Bidirectional(
        #     layers.LSTM(64, dropout = 0.5, return_sequences = True, return_state = True)
        # )(lstm)
        #
        # state_h = layers.Concatenate()([forward_h, backward_h])
        # state_c = layers.Concatenate()([forward_c, backward_c])
        #
        # attention = layers.Attention()
        # context_vector, attention_weights = attention(lstm, state_h)

        attention = Attention(return_squences = False)(lstm)
        dropout = layers.Dropout(0.5)(attention)

        dense = layers.Dense(20, activation = 'relu')(dropout)
        dropout = layers.Dropout(0.5)(dense)
        output = layers.Dense(1, activation = 'sigmoid')(dropout)

        super().__init__(input, output)

        self.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

def main() :
    vocab_size = 10000
    max_len = 500

    data = DataSet()
    model = Model(max_len, vocab_size)

    history = model.fit(data.X_train, data.y_train, epochs = 5, batch_size = 256, validation_split = 0.1, verbose = 1)
    print("\n 테스트 정확도 : %.4f" % (model.evaluate(data.X_test, data.y_test)[1]))

if __name__ == '__main__' :
    main()