# Import lbraries
import numpy as np
from save_and_load_data_class import *
from preprocess_data import Data
import matplotlib.pyplot as plt
import keras
from sklearn.metrics import roc_auc_score

if __name__ == "__main__":

    # Load data
    base_path = "C:\\Users\\\matth\\PycharmProjects\\Data_for_all_projects\\NLP_toxic_comments"
    data = Data(path=base_path)
    save_class_as_pickle(path=base_path, Obj=data)
    # data = load_pickle(path=base_path, class_name="data")

    # Build model
    print("Building model...")
    input_ = keras.layers.Input(shape=(data.max_sequence_length,))
    x = keras.layers.Embedding(input_dim=data.embedding_matrix.shape[0],
                               output_dim=data.embedding_dim,
                               weights=[data.embedding_matrix],
                               input_length=data.max_sequence_length,
                               trainable=False)(input_)
    x = keras.layers.Bidirectional(keras.layers.LSTM(15, return_sequences=True))(x)
    x = keras.layers.GlobalMaxPool1D()(x)
    output = keras.layers.Dense(6, activation="sigmoid")(x)

    model = keras.models.Model(input_, output)
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.01), metrics=['accuracy'])

    # Train model
    print('Training model...')
    r = model.fit(
        data.padded_sequences,
        data.labels,
        batch_size=128,
        epochs=1,
        validation_split=0.2
    )

    # plot some data
    plt.plot(r.history['loss'], label='loss')
    plt.plot(r.history['val_loss'], label='val_loss')
    plt.legend()
    plt.show()

    # accuracies
    plt.plot(r.history['accuracy'], label='acc')
    plt.plot(r.history['val_accuracy'], label='val_acc')
    plt.legend()
    plt.show()

    p = model.predict(data.padded_sequences)
    aucs = []
    for j in range(6):
        auc = roc_auc_score(data.labels[:, j], p[:, j])
        aucs.append(auc)
    print(np.mean(aucs))

