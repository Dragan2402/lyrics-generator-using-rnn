# Import the dependencies
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, Flatten, Dense, CuDNNLSTM
from keras.models import Sequential
from keras.utils import np_utils

dataset = pd.read_csv('data/taylor_swift_lyrics.csv', encoding="latin1")

os.system('cls')
n = input("Generate text\n\t1-yes\n\tElse-no\n\t")


# PROCESS CSV DATA INTO RAW SONGS
def process_first_line(lyrics_p, song_id_p, song_name_p, row_p):
    lyrics_p.append(row_p['lyric'] + '\n')
    song_id_p.append(row_p['year'] * 100 + row_p['track_n'])
    song_name_p.append(row_p['track_title'])
    return lyrics_p, song_id_p, song_name_p


if n == "1":
    lyrics = []
    songID = []
    songName = []
    songNumber = 1
    i = 0
    isFirstLine = True
    for index, row in dataset.iterrows():
        if songNumber == row['track_n']:
            if isFirstLine:
                lyrics, songID, songName = process_first_line(lyrics, songID, songName, row)
                isFirstLine = False
            else:
                lyrics[i] += row['lyric'] + '\n'
        else:
            lyrics, songID, songName = process_first_line(lyrics, songID, songName, row)
            songNumber = row['track_n']
            i += 1
    lyrics = []
    songID = []
    songName = []
    songNumber = 1
    i = 0
    isFirstLine = True
    for index, row in dataset.iterrows():
        if songNumber == row['track_n']:
            if isFirstLine:
                lyrics, songID, songName = process_first_line(lyrics, songID, songName, row)
                isFirstLine = False
            else:
                lyrics[i] += row['lyric'] + '\n'
        else:
            lyrics, songID, songName = process_first_line(lyrics, songID, songName, row)
            songNumber = row['track_n']
            i += 1
    lyrics_data = pd.DataFrame({'songID': songID, 'songName': songName, 'lyrics': lyrics})
    training_number = round(i * 0.8)
    validation_number = i - training_number
    with open('data/lyricsText.txt', 'w', encoding="utf-8") as filehandle:
        for listitem in lyrics:
            filehandle.write('%s\n' % listitem)
    with open('data/lyricsText_validation.txt', 'w', encoding="utf-8") as filehandle:
        for listitem in lyrics[training_number:]:
            filehandle.write('%s\n' % listitem)

# LOAD SONGS
train_text_file_name = 'data/lyricsText.txt'

raw_text = open(train_text_file_name, encoding='UTF-8').read()
raw_text = raw_text.lower()

chars = sorted(list(set(raw_text)))
int_chars = dict((i, c) for i, c in enumerate(chars))
chars_int = dict((i, c) for c, i in enumerate(chars))

n_chars = len(raw_text)
n_vocab = len(chars)

# DISPLAY DATA
to_display_statistics = input("Display data statistics\n\t1-yes\n\tElse-no\n\t")
if to_display_statistics == "1":
    print(dataset.head())
    print(dataset.describe())
    print('Total Characters : ', n_chars)  # number of all the characters in lyricsText.txt
    print('Total Vocab : ', n_vocab)  # number of unique characters

seq_len = 100
data_X = []
data_y = []

for i in range(0, n_chars - seq_len, 1):
    seq_in = raw_text[i:i + seq_len]
    seq_out = raw_text[i + seq_len]
    data_X.append([chars_int[char] for char in seq_in])
    data_y.append(chars_int[seq_out])
n_patterns = len(data_X)

if to_display_statistics == "1":
    print('Total Patterns : ', n_patterns)

X = np.reshape(data_X, (n_patterns, seq_len, 1))
X = X / float(n_vocab)
y = np_utils.to_categorical(data_y)

# MODEL CONFIG
LSTM_layer_num = 4
layer_size = [256, 256, 256, 256]
model = Sequential()
model.add(CuDNNLSTM(layer_size[0], input_shape=(X.shape[1], X.shape[2]), return_sequences=True))

for i in range(1, LSTM_layer_num):
    model.add(CuDNNLSTM(layer_size[i], return_sequences=True))

model.add(Flatten())
model.add(Dense(y.shape[1]))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

model.summary()

checkpoint_name = 'weights/Weights-LSTM-improvement-{epoch:03d}-{loss:.5f}-bigger.hdf5'
checkpoint = ModelCheckpoint(checkpoint_name, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# Fit the model :
model_params = {'epochs': 100,
                'batch_size': 128,
                'callbacks': callbacks_list,
                'verbose': 1,
                'validation_split': 0.2,
                'validation_data': None,
                'shuffle': True,
                'initial_epoch': 0,
                'steps_per_epoch': None,
                'validation_steps': None}

# TRAIN MODEL
train_model = input("\nTrain model\n\t1-yes\n\tElse-no\n\t")
if train_model == "1":
    history = model.fit(X,
                        y,
                        epochs=model_params['epochs'],
                        batch_size=model_params['batch_size'],
                        callbacks=model_params['callbacks'],
                        verbose=model_params['verbose'],
                        validation_split=model_params['validation_split'],
                        validation_data=model_params['validation_data'],
                        shuffle=model_params['shuffle'],
                        initial_epoch=model_params['initial_epoch'],
                        steps_per_epoch=model_params['steps_per_epoch'],
                        validation_steps=model_params['validation_steps'])
    history_df = pd.DataFrame(history.history)
    data = history_df["loss"]
    epochs = []
    epoch_num = 1
    values = []
    for single_data in data:
        epochs.append(epoch_num)
        values.append(single_data)
        epoch_num += 1
    fig = plt.figure(figsize=(15, 4), facecolor="#B291B6")
    plt.plot(epochs, values, label="loss plot")

    plt.show()
    print(data)
# LOAD WEIGHTS
else:
    wights_file = 'weights/Weights-LSTM-improvement-099-0.01464-bigger.hdf5'
    model.load_weights(wights_file)

model.compile(loss='categorical_crossentropy', optimizer='adam')
start = np.random.randint(0, len(data_X) - 1)
pattern = data_X[start]
print('\nSeed : ')
print("\"", ''.join([int_chars[value] for value in pattern]), "\"\n")

generated_characters = int(input("\nNumber of characters:\n"))
text = ''
# GENERATE SONG
for i in range(generated_characters):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = int_chars[index]
    sys.stdout.write(result)
    text += result
    pattern.append(index)
    pattern = pattern[1:len(pattern)]

filename = input("\nEnter file name to save the song:\n")
filename = "results/" + filename + ".txt"
text_file = open(filename, "w")
text_file.write(text)
text_file.close()

print('\nDone')
input()
