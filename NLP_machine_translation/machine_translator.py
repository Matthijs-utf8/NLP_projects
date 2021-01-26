""" A large part of this code was inspired by the work of LazyProgrammer"""

# Import libraries
import numpy as np
from save_and_load_data_class import *
from preprocess_data import Data
import matplotlib.pyplot as plt
import keras
from keras.layers import Input, LSTM, Dense, Embedding
from keras.models import Model
import keras.backend as K

if __name__ == "__main__":

    # Load data
    base_path = "C:\\Users\\matth\\PycharmProjects\\Data_for_all_projects\\NLP_machine_translation"
    # data = Data(path=base_path)
    # save_class_as_pickle(path=base_path, Obj=data)
    data = load_pickle(path=base_path, class_name="data")

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
    encoder_lstm = LSTM(units=LATENT_DIM, # LSTM consists of LATENT_DIM units in a row
                        return_state=True) # Returns not only the last output, but also the last hidden- and cell state

    ### DECODER LAYERS ###
    decoder_input = Input(shape=(data.decoder_inputs.shape[1],))
    decoder_embedding = Embedding(input_dim=data.num_words_outputs, # The input dimension is the same as the size of the dictionary
                                  output_dim=data.embedding_dim) # For each word we output a feature vector
    decoder_lstm = LSTM(units=LATENT_DIM, # LSTM consists of LATENT_DIM units in a row
                        return_state=True, # Returns not only the last output, but also the last hidden- and cell state
                        return_sequences=True) # Since the decoder is a "to-many" model: return_sequences=True
    decoder_dense = Dense(units=data.num_words_outputs, # Return a prediction value for each word in the output dictionary
                          activation='softmax')  # Use softmax, as we are working with one-hot encoded words

    ### MODEL PIPELINE ###
    x = encoder_embedding(encoder_input)  # Pass the input through embedding layer
    encoder_output, hidden_state_encoder_output, cell_state_encoder_output = encoder_lstm(x)  # Get the encoder output and states
    x = decoder_embedding(decoder_input) # Tell the decoder what the word should have been through teacher-forcing
    decoder_outputs, _, _ = decoder_lstm(x, initial_state=[hidden_state_encoder_output, cell_state_encoder_output]) # The decoder starts with the last hidden- and cell state from decoder.
    decoder_outputs = decoder_dense(decoder_outputs)

    ### CREATE MODEL ###
    model = Model([encoder_input, decoder_input], # Inputs
                  decoder_outputs) # Outputs
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
