""" A large part of this code was inspired by the work of LazyProgrammer"""

# Import lbraries
import numpy as np
import os
import keras
import traceback

class Data:

    def __init__(self, path):
        """
        Initialize Data object
        :param path: str,   the path should be the base path for the pretrained word embeddings, train- and test csv
        """

        # Get the class name
        (filename, line_number, function_name, text) = traceback.extract_stack()[-2]
        self.name = text[:text.find('=')].strip()

        # Configuration
        self.num_samples = 10000
        self.max_vocab_size = 3000
        self.embedding_dim = int(100)
        self.path = path

        self.prepare_data()  # Tokenize the sentences to sequences of indexes
        self.make_embedding_matrix()  # Load the pretrained embedding matrix

        # Initiate empty array
        self.decoder_one_hot_targets = np.zeros((len(self.input_sequences), self.max_sequence_length_outputs, self.num_words_outputs), dtype='float32')
        for i, target_sequence in enumerate(self.decoder_targets):
            for t, word in enumerate(target_sequence):
                if word > 0: # If it is not padding
                    self.decoder_one_hot_targets[i, t, word] = 1 # One-hot encode the each word in the sentence by setting its index value to 1 and leaving the rest at 0

    def prepare_data(self):

        # Prepare text samples and their labels
        print('Loading poetry...')
        input_texts = []  # sentence in original language
        target_texts = []  # sentence in target language
        target_texts_inputs = []  # sentence in target language offset by 1

        # Loop through all lines in the data up to the "self.samples"th line
        for n, line in enumerate(open(self.path + '\\nld.txt', "rb")):

            # Decode the line
            line = line.decode()

            # Only keep a limited number of samples.
            if n > self.num_samples:
                break

            # Input and target are separated by tab. Check if the line has a tab. If it hasn't: continue.
            if '\t' not in line:
                continue

            # Split up the input and translation.
            input_line, translation, *rest = line.rstrip().split('\t')

            target_line = translation + ' <eos>'
            target_line_input = '<sos> ' + translation

            input_texts.append(input_line)
            target_texts.append(target_line)
            target_texts_inputs.append(target_line_input)

        print("num samples:", len(input_texts))
        self.input_texts = input_texts

        # Convert the input sentences (strings) into integers
        tokenizer_inputs = keras.preprocessing.text.Tokenizer(num_words=self.max_vocab_size) # Initialize tokenizer
        tokenizer_inputs.fit_on_texts(input_texts)
        self.input_sequences = tokenizer_inputs.texts_to_sequences(input_texts)
        self.word2index_inputs = tokenizer_inputs.word_index
        self.num_words_inputs = min(self.max_vocab_size, len(self.word2index_inputs) + 1)
        self.max_sequence_length_inputs = max(len(s) for s in self.input_sequences)
        self.encoder_inputs = keras.preprocessing.sequence.pad_sequences(self.input_sequences, maxlen=self.max_sequence_length_inputs)  # Pad all sequences to the same length by padding them with zeros (at the front of the list) until the sequence length equals the max sequence length
        print("encoder_inputs.shape:", self.encoder_inputs.shape)
        print("encoder_inputs[0]:", self.encoder_inputs[0])

        # Convert the input sentences (strings) into integers
        tokenizer_outputs = keras.preprocessing.text.Tokenizer(num_words=self.max_vocab_size, filters="") # Initialize tokenizer
        tokenizer_outputs.fit_on_texts(target_texts + target_texts_inputs)
        self.target_sequences = tokenizer_outputs.texts_to_sequences(target_texts)
        self.target_sequences_inputs = tokenizer_outputs.texts_to_sequences(target_texts_inputs)
        self.word2index_outputs = tokenizer_outputs.word_index
        assert ('<sos>' in self.word2index_outputs)
        assert ('<eos>' in self.word2index_outputs)
        self.num_words_outputs = len(self.word2index_outputs) + 1
        self.max_sequence_length_outputs = max(len(s) for s in self.target_sequences)
        self.decoder_inputs = keras.preprocessing.sequence.pad_sequences(self.target_sequences_inputs, maxlen=self.max_sequence_length_outputs, padding='post')
        print("decoder_inputs.shape:", self.decoder_inputs.shape)
        print("decoder_inputs[0]:", self.decoder_inputs[0])

        self.decoder_targets = keras.preprocessing.sequence.pad_sequences(self.target_sequences, maxlen=self.max_sequence_length_outputs, padding='post')

    def make_embedding_matrix(self):

        # Load word-vector embedding
        print("Loading word vectors...")
        self.word2vec = {}
        with open(self.path + "\\glove.6B.{}d.txt".format(self.embedding_dim), "rb") as f: # This text file contains pre-loaded word embeddings
            # Word embedding is just saved as a space-separated text file in the format: word vec[0] vec[1] vec[2] ...
            for line in f:
                line = line.decode() # Decode the bytes in the line before reading
                values = line.split()  # This converts every line to [word, vec[0], vec[1], vec[2], ...]
                word = values[0]  # Get the word, which is at the zeroth position
                vec = np.asarray(values[1:], dtype="float32")  # Save the embedding values in an array
                self.word2vec[word] = vec  # Put the whole thing in a dictionary like: {"word": nparray([vec[0], vec[1], vec[2], ...]), word2: ...}
        print('Found %s word vectors.' % len(self.word2vec))

        # Prepare embedding matrix
        print('Filling pre-trained embeddings...')
        num_words = int(min(self.max_vocab_size, len(self.word2index_inputs) + 1))  # Get the most common words, or just all words if there are less than the maximum
        self.embedding_matrix = np.zeros((num_words, self.embedding_dim))  # Initialize empty embedding matrix
        for word, i in self.word2index_inputs.items():  # Loop through word2index dictionary
            if i < self.max_vocab_size:  # If the word is not over the ith index
                embedding_vector = self.word2vec.get(word)  # Get the vector that accompanies the word
                if embedding_vector is not None:  # If the word is found in the pretrained word embedding, else the vector will be all zeros
                    self.embedding_matrix[i] = embedding_vector # !!!!!!!! This puts the first word at index 1 !!!!!!!!!!