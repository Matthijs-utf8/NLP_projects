""" A large part of this code was inspired by the work of LazyProgrammer"""

# Import libraries
import numpy as np
from NLP_machine_translation.save_and_load_data_class import *
from NLP_machine_translation.preprocess_data import Data
import matplotlib.pyplot as plt
import keras
from keras.layers import Input, LSTM, GRU, Dense, Embedding, \
  Bidirectional, RepeatVector, Concatenate, Activation, Dot, Lambda
from keras.models import Model
import keras.backend as K

# make sure we do softmax over the time axis
# expected shape is N x T x D
# note: the latest version of Keras allows you to pass in axis arg
def softmax_over_time(x):
  assert(K.ndim(x) > 2)
  e = K.exp(x - K.max(x, axis=1, keepdims=True))
  s = K.sum(e, axis=1, keepdims=True)
  return e / s

if __name__ == "__main__":

    # Load data
    base_path = "C:\\Users\\matth\\PycharmProjects\\Data_for_all_projects\\NLP_machine_translation"
    data = Data(path=base_path)
    save_class_as_pickle(path=base_path, Obj=data)
    # data = load_pickle(path=base_path, class_name="data")

    # Training configuration
    VALIDATION_SPLIT = 0.2  # Split of validation set
    BATCH_SIZE = 64  # Batch size for training.
    EPOCHS = 20  # Number of epochs to train for.
    LATENT_DIM = 256  # Latent dimensionality of the encoding space.

    print("Building prediction model...")

    ### ENCODER LAYERS ###
    encoder_input = keras.layers.Input(shape=(data.encoder_inputs.shape[1],)) # Input is a sequence
    encoder_embedding = Embedding(input_dim=data.num_words_inputs, # The input dimension is the same as the size of the dictionary
                                  output_dim=data.embedding_dim, # For each word we output a feature vector
                                  weights=[data.embedding_matrix], # Add the pre-trained weights
                                  trainable=False) # Don't train on pre-trained weights
    encoder_lstm = Bidirectional(LSTM(units=LATENT_DIM, # LSTM consists of LATENT_DIM units in a row
                                      return_state=True)) # Returns not only the last output, but also the last hidden- and cell state

    ### ATTENTION ###
    # Attention layers need to be global because they will be repeated Ty times at the decoder
    attn_repeat_layer = RepeatVector(data.max_sequence_length_inputs)
    attn_concat_layer = Concatenate(axis=-1)  #
    attn_dense1 = Dense(10, activation='tanh')
    attn_dense2 = Dense(1, activation=keras.activations.softmax(axis=1))
    attn_dot = Dot(axes=1)  # to perform the weighted sum of alpha[t] * h[t]

    def one_step_attention(h, st_1):
        # h = h(1), ..., h(Tx), shape = (Tx, LATENT_DIM * 2) --> encoder hidden states
        # st_1 = s(t-1), shape = (LATENT_DIM_DECODER,) --> last output

        # Copy s(t-1) Tx times now shape = (Tx, LATENT_DIM_DECODER)
        st_1 = attn_repeat_layer(st_1)

        # Concatenate all h(t)'s with s(t-1)
        # Now of shape (Tx, LATENT_DIM_DECODER + LATENT_DIM * 2)
        x = attn_concat_layer([h, st_1])

        # Neural net first layer
        x = attn_dense1(x)

        # Neural net second layer with special softmax over time
        alphas = attn_dense2(x)

        # "Dot" the alphas and the h's
        # Remember a.dot(b) = sum over a[t] * b[t]
        context = attn_dot([alphas, h])

        return context

    ### DECODER LAYERS ###
    initial_s = Input(shape=(LATENT_DIM,), name='s0')  # The initial hidden state needs to be inputted
    initial_c = Input(shape=(LATENT_DIM,), name='c0')  # The initial cell state needs to be inputted
    decoder_input = Input(shape=(data.decoder_inputs.shape[1],)) # Input the correct word with teacher forcing
    decoder_embedding = Embedding(input_dim=data.num_words_outputs, # The input dimension is the same as the size of the dictionary
                                  output_dim=data.embedding_dim) # For each word we output a feature vector
    decoder_lstm = LSTM(units=LATENT_DIM, # LSTM consists of LATENT_DIM units in a row
                        return_state=True, # Returns not only the last output, but also the last hidden- and cell state
                        return_sequences=True) # Since the decoder is a "to-many" model: return_sequences=True
    decoder_dense = Dense(units=data.num_words_outputs, # Return a prediction value for each word in the output dictionary
                          activation='softmax')  # Use softmax, as we are working with one-hot encoded words
    context_last_word_concat_layer = Concatenate(axis=2) # Concatenate the last word with ne new context

    ### MODEL PIPELINE ###
    x = encoder_embedding(encoder_input)  # Pass the input through embedding layer
    encoder_output, hidden_state_encoder_output, cell_state_encoder_output = encoder_lstm(x)  # Get the encoder output and states
    deconder_embedding_output = decoder_embedding(decoder_input) # Tell the decoder what the word should have been through teacher-forcing

    # Unlike previous seq2seq, we cannot get the output all in one step
    # Instead we need to do Ty steps
    # In each of those steps, we need to consider all Tx h's

    # s, c will be re-assigned in each iteration of the loop
    s = initial_s
    c = initial_c
    outputs = [] #  Collect outputs in a list

    for t in range(data.max_len_target):  # Ty times

        # Get the context using attention
        context = one_step_attention(encoder_output, s)

        # We need a different layer for each time step
        selector = Lambda(lambda x: x[:, t:t + 1])
        xt = selector(deconder_embedding_output)

        # Combine
        decoder_lstm_input = context_last_word_concat_layer([context, xt])

        # Pass the combined [context, last word] into the LSTM along with [s, c]
        o, s, c = decoder_lstm(decoder_lstm_input, initial_state=[s, c]) # Get the new [s, c] and output

        # Final dense layer to get next word prediction
        decoder_output = decoder_dense(o)
        outputs.append(decoder_output)


    # 'outputs' is now a list of length Ty
    # each element is of shape (batch size, output vocab size)
    # therefore if we simply stack all the outputs into 1 tensor
    # it would be of shape T x N x D
    # we would like it to be of shape N x T x D

    def stack_and_transpose(x):
        # x is a list of length T, each element is a batch_size x output_vocab_size tensor
        x = K.stack(x)  # is now T x batch_size x output_vocab_size tensor
        x = K.permute_dimensions(x, pattern=(1, 0, 2))  # is now batch_size x T x output_vocab_size
        return x

    # Make it a layer
    stacker = Lambda(stack_and_transpose)
    outputs = stacker(outputs)

    ### CREATE MODEL ###
    model = Model(inputs=[encoder_input, decoder_input, initial_s, initial_c], outputs=outputs)

    model.compile(optimizer="adam",
                  loss="categorical-crossentropy", # Use categorical cross-entropy for one-hot encoded outputs
                  metrics=["accuracy"])

    ### FIT AND SAVE MODEL ###
    r = model.fit([data.encoder_inputs, data.decoder_inputs],
                  data.decoder_one_hot_targets,
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS,
                  validation_split=VALIDATION_SPLIT)

    model.save(base_path + '\\model1.h5')

    ### PLOT LOSS ###
    plt.plot(r.history['loss'], label='loss')
    plt.plot(r.history['val_loss'], label='val_loss')
    plt.legend()
    plt.show()

    print("Building sampling model...")

    ### SAMPLING ENCODER ###
    encoder_model = keras.models.Model(encoder_input, # Input: y(t-1)
                                       [hidden_state_encoder_output, cell_state_encoder_output]) # Output: h(t-1), c(t-1)

    ### SAMPLING DECODER ###
    decoder_inputs_single = keras.layers.Input(shape=(1,)) # Decoder receives one word at a time
    decoder_states_inputs = [keras.layers.Input(shape=(LATENT_DIM,)), keras.layers.Input(shape=(LATENT_DIM,))] # Decoder also receives the last hidden- and cell state from the encoder
    decoder_outputs, h, c = decoder_lstm(decoder_embedding(decoder_inputs_single), initial_state=decoder_states_inputs)
    decoder_states = [h, c]
    decoder_outputs = decoder_dense(decoder_outputs)

    ### CREATE SAMPLING MODEL ###
    decoder_model = keras.models.Model([decoder_inputs_single] + decoder_states_inputs, # Input: y(t-1), h(t-1), c(t-1)
                                       [decoder_outputs] + decoder_states) # Output: y(t), h(t), c(t)

    ### GET INDEX TO WORD DICTIONARIES ###
    idx2word_eng = {v: k for k, v in data.word2index_inputs.items()}
    idx2word_trans = {v: k for k, v in data.word2index_outputs.items()}

    ### DECODE SEQUENCE ###
    def decode_sequence(input_seq):

        # Encode the input as state vectors.
        states_value = encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1))

        # Populate the first character of target sequence with the start character.
        target_seq[0, 0] = data.word2index_outputs['<sos>']

        # Create the translation
        output_sentence = []
        for _ in range(data.max_sequence_length_outputs):

            # Get the output and new hidden- and cell state
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

            # Get next word
            idx = np.argmax(output_tokens[0, 0, :]) # Choose the word with highest probability

            # End sentence of EOS
            if idx == data.word2index_outputs['<eos>']:
                break

            word = ''
            if idx > 0:
                word = idx2word_trans[idx]
                output_sentence.append(word)

            # Update the decoder input, which is just the word just generated
            target_seq[0, 0] = idx

            # Update states
            states_value = [h, c]

        return ' '.join(output_sentence)


    while True:

        # Do some test translations
        i = np.random.choice(len(data.input_texts))
        input_seq = data.encoder_inputs[i:i + 1]
        translation = decode_sequence(input_seq)
        print('-')
        print('Input:', data.input_texts[i])
        print('Translation:', translation)

        ans = input("Continue? [Y/n]")
        if ans and ans.lower().startswith('n'):
            break
