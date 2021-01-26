# Import lbraries
import numpy as np
import pandas as pd
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
        self.max_vocab_size = 2e4
        self.max_sequence_length = 100
        self.embedding_dim = int(100)
        self.path = path

        self.prepare_data()  # Tokenize the sentences to sequences of indexes
        self.make_embedding_matrix()  # Load the pretrained embedding matrix

    def prepare_data(self):

        # Prepare text samples and their labels
        print('Loading comments...')
        df = pd.read_csv(self.path + "\\train.csv")  # Load csv
        possible_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]  # Gets the possible labels

        sentences = df["comment_text"].fillna("DUMMY_VALUE")  # Fills any rows that do not contain a sentence (which is none)
        self.labels = df[possible_classes].values  # Get the valued labels for each comment as numpy array

        # Convert the sentences (strings) into integers
        tokenizer = keras.preprocessing.text.Tokenizer(num_words=self.max_vocab_size)  # Initialize tokenizer
        tokenizer.fit_on_texts(sentences) # Updates internal vocabulary based on a list of sentences.

        self.word2index = tokenizer.word_index # Save the dictionary that tells us the corresponding index of each word
        self.sequences = tokenizer.texts_to_sequences(sentences) # Returns a number for each sentence that corresponds with keras' internal library and puts those number is a list
        self.padded_sequences = keras.preprocessing.sequence.pad_sequences(self.sequences, maxlen=self.max_sequence_length)     # Pad all sequences to the same length by padding them with zeros (at the front of the list) until the sequence length equals the max sequence length

    def make_embedding_matrix(self):

        # Load word-vector embedding
        print("Loading word vectors...")
        self.word2vec = {}
        with open(self.path + "\\glove.6B.{}d.txt".format(self.embedding_dim), "rb") as f:
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