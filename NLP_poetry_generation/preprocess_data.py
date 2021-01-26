
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
        self.max_vocab_size = 3000
        self.max_sequence_length = 100
        self.embedding_dim = int(50)
        self.path = path

        self.prepare_data()  # Tokenize the sentences to sequences of indexes
        self.make_embedding_matrix()  # Load the pretrained embedding matrix

        # Initiate empty array
        self.one_hot_targets = np.zeros((len(self.input_sequences), self.max_sequence_length, self.num_words))
        for i, target_sequence in enumerate(self.target_sequences):
            for t, word in enumerate(target_sequence):
                if word > 0:
                    self.one_hot_targets[i, t, word] = 1 # One-hot encode the each word in the sentence by setting its index value to 1 and leaving the rest at 0

    def prepare_data(self):

        # Prepare text samples and their labels
        print('Loading poetry...')
        input_texts = []
        target_texts = []
        for line in open(self.path + '\\robert_frost.txt', "rb"):
            line = line.rstrip()
            if not line:
                continue

            input_line = '<sos> ' + line
            target_line = line + ' <eos>'

            input_texts.append(input_line)
            target_texts.append(target_line)

        all_lines = input_texts + target_texts

        # Convert the sentences (strings) into integers
        tokenizer = keras.preprocessing.text.Tokenizer(num_words=self.max_vocab_size, filters="")  # Initialize tokenizer, don't use a filter because of our <SOS> and <EOS>
        tokenizer.fit_on_texts(all_lines) # Updates internal vocabulary based on a list of sentences.

        self.word2index = tokenizer.word_index # Save the dictionary that tells us the corresponding index of each word
        self.num_words = min(self.max_vocab_size, len(self.word2index) + 1)
        assert ('<sos>' in self.word2index)
        assert ('<eos>' in self.word2index)
        self.input_sequences = tokenizer.texts_to_sequences(input_texts) # Returns a number for each word in the sentence that corresponds with keras' internal library and puts those numbers in a list
        self.target_sequences = tokenizer.texts_to_sequences(target_texts) # Returns a number for each word in the sentence that corresponds with keras' internal library and puts those numbers in a list
        max_sequence_length_from_data = max(len(s) for s in self.input_sequences)
        self.max_sequence_length = min(max_sequence_length_from_data, self.max_sequence_length)
        self.input_padded_sequences = keras.preprocessing.sequence.pad_sequences(self.input_sequences, maxlen=self.max_sequence_length, padding="post")     # Pad all sequences to the same length by padding them with zeros (at the front of the list) until the sequence length equals the max sequence length
        self.target_padded_sequences = keras.preprocessing.sequence.pad_sequences(self.target_sequences, maxlen=self.max_sequence_length, padding="post")     # Pad all sequences to the same length by padding them with zeros (at the front of the list) until the sequence length equals the max sequence length

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
        num_words = int(min(self.max_vocab_size, len(self.word2index) + 1))  # Get the most common words, or just all words if there are less than the maximum
        self.embedding_matrix = np.zeros((num_words, self.embedding_dim))  # Initialize empty embedding matrix
        for word, i in self.word2index.items():  # Loop through word2index dictionary
            if i < self.max_vocab_size:  # If the word is not over the ith index
                embedding_vector = self.word2vec.get(word)  # Get the vector that accompanies the word
                if embedding_vector is not None:  # If the word is found in the pretrained word embedding, else the vector will be all zeros
                    self.embedding_matrix[i] = embedding_vector # !!!!!!!! This puts the first word at index 1 !!!!!!!!!!