

# Import libraries
import numpy as np
from save_and_load_data_class import *
from preprocess_data import Data
import matplotlib.pyplot as plt
import keras


if __name__ == "__main__":

    # Load data
    base_path = "C:\\Users\\\matth\\PycharmProjects\\Data_for_all_projects\\NLP_poetry_generation"
    data = Data(path=base_path)
    save_class_as_pickle(path=base_path, Obj=data)
    # data = load_pickle(path=base_path, class_name="data")

    # Set training configuration
    VALIDATION_SPLIT = 0.2
    BATCH_SIZE = 128
    EPOCHS = 2000
    LATENT_DIM = 25
    initial_state = np.zeros((len(data.input_padded_sequences), LATENT_DIM))

    print('Building word prediction model...')
    # Input layers
    input_ = keras.layers.Input(shape=(data.max_sequence_length,)) # Create an input layer for the poetry data
    initial_h = keras.layers.Input(shape=(LATENT_DIM,)) # Make the initial hidden state of the LSTM an input, so sentence the model starts in the same state
    initial_c = keras.layers.Input(shape=(LATENT_DIM,)) # Make the initial cell  state of the LSTM an input, so sentence the model starts in the same state

    # Embedding layer
    embedding_layer = keras.layers.Embedding(data.num_words, data.embedding_dim, weights=[data.embedding_matrix]) # Load pre-trained word embeddings into an embedding layer
    x = embedding_layer(input_) # Input the first input layer to the embedding layer

    # LSTM layer
    lstm = keras.layers.LSTM(LATENT_DIM, return_sequences=True, return_state=True)
    x, _, _ = lstm(x, initial_state=[initial_h, initial_c]) # Get the output of the LSTM, without the hidden state and cell state

    # Dense layer
    dense = keras.layers.Dense(data.num_words, activation='softmax') # Return the output as a one-hot encoding of each next word in the sentence
    output = dense(x)

    model = keras.models.Model([input_, initial_h, initial_c], output) # Construct model from specified layers
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.01), metrics=['accuracy']) # Compile the model

    print('Training word prediction model...')
    # Train the model to output the next word in the sentence based on the poetry of Robert Frost
    r = model.fit([data.input_padded_sequences, initial_state, initial_state], output=data.one_hot_targets, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=VALIDATION_SPLIT)

    # Plot loss
    plt.plot(r.history['loss'], label='loss')
    plt.plot(r.history['val_loss'], label='val_loss')
    plt.legend()
    plt.show()

    # Plot accuracy
    plt.plot(r.history['accuracy'], label='acc')
    plt.plot(r.history['val_accuracy'], label='val_acc')
    plt.legend()
    plt.show()

    print('Building sampling model...')
    input2 = keras.layers.Input(shape=(1,))  # Only input one word at a time
    x = embedding_layer(input2) # Embedding layer is the same as in the prediction model
    x, h, c = lstm(x, initial_state=[initial_h, initial_c]) # Build the same LSTM layer as before, but now output the hidden- and cell-state as well
    output2 = dense(x) # Output a one-hot encoded array
    sampling_model = keras.models.Model([input2, initial_h, initial_c], [output2, h, c])

    # reverse word2idx dictionary to get back words during prediction
    idx2word = {v: k for k, v in data.word2index.items()}


    def sample_line():

        # Initial inputs
        np_input = np.array([[data.word2index['<sos>']]]) # Shape = (1, 1)
        h = np.zeros((1, LATENT_DIM)) # Start with the same hidden state as the first model was trained at (hidden state all zeros)
        c = np.zeros((1, LATENT_DIM)) # Start with the same cell state as the first model was trained at (cell state all zeros)

        # Initiate a list to store the output sentence
        output_sentence = []

        for _ in range(data.max_sequence_length): # Generate words until the maximum sentence length is hit

            # Get the predicted next word as well as the next hidden- and cell-state
            # The next word, "o", is now outputted as a one-hot encoded word (actually a probability distribution for all the words)
            o, h, c = sampling_model.predict([np_input, h, c])

            # Sometimes the model gives a really high probability to index 0, but no word has index 0 in the tokenizer. This deals with that error.
            probs = o[0, 0]
            if np.argmax(probs) == 0:
                print("wtf")
            probs[0] = 0
            probs /= probs.sum()

            # Choose from the probability distribution
            idx = np.random.choice(len(probs), p=probs)

            # Stop generating words if the end of sentence is hit
            if idx == data.word2index['<eos>']:
                break

            # Calculate output
            output_sentence.append(idx2word.get(idx, '<WTF %s>' % idx))

            # Create the next input into model
            np_input[0, 0] = idx # Set the input to the predicted word (the input is the index of the word)

        return ' '.join(output_sentence)


    # generate a 4 line poem
    while True:
        for _ in range(4):
            print(sample_line())

        ans = input("---generate another? [Y/n]---")
        if ans and ans[0].lower().startswith('n'):
            break