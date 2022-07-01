from textgenrnn import textgenrnn
import os
import pandas as pd

model_cfg = {
    'rnn_layers': 12,
    'rnn_bidirectional': True,
    'max_length': 15,
    'max_words': 10000,
    'dim_embeddings': 100,
    'word_level': False,
}

train_cfg = {
    'line_delimited': True,
    'num_epochs': 100,
    'gen_epochs': 25,
    'batch_size': 750,
    'train_size': 0.8,
    'dropout': 0.0,
    'max_gen_length': 300,
    'validation': True,
    'is_csv': False
}

dataset = pd.read_csv('data/taylor_swift_lyrics.csv', encoding="latin1")

os.system('cls')
n = input("Generate text\n\t1-yes\n\tElse-no\n\t")


def process_first_line(lyrics_p, song_id_p, song_name_p, row_p):
    lyrics_p.append(row_p['lyric'] + '\n')
    song_id_p.append(row_p['year'] * 100 + row_p['track_n'])
    song_name_p.append(row_p['track_title'])
    return lyrics_p, song_id_p, song_name_p


if n == "1":

    # define empty lists for the lyrics , songID , songName
    lyrics = []
    songID = []
    songName = []

    # songNumber indicates the song number in the dataset
    songNumber = 1

    # i indicates the song number
    i = 0
    isFirstLine = True

    # Iterate through every lyrics line and join them together for each song independently
    for index, row in dataset.iterrows():
        if songNumber == row['track_n']:
            if isFirstLine:
                lyrics, songID, songName = process_first_line(lyrics, songID, songName, row)
                isFirstLine = False
            else:
                # if we still in the same song , keep joining the lyrics lines
                lyrics[i] += row['lyric'] + '\n'
        # When it's done joining a song's lyrics lines , go to the next song :
        else:
            lyrics, songID, songName = process_first_line(lyrics, songID, songName, row)
            songNumber = row['track_n']
            i += 1

    # define empty lists for the lyrics , songID , songName
    lyrics = []
    songID = []
    songName = []

    # songNumber indicates the song number in the dataset
    songNumber = 1

    # i indicates the song number
    i = 0
    isFirstLine = True

    # Iterate through every lyrics line and join them together for each song independently
    for index, row in dataset.iterrows():
        if songNumber == row['track_n']:
            if isFirstLine:
                lyrics, songID, songName = process_first_line(lyrics, songID, songName, row)
                isFirstLine = False
            else:
                # if we still in the same song , keep joining the lyrics lines
                lyrics[i] += row['lyric'] + '\n'
        # When it's done joining a song's lyrics lines , go to the next song :
        else:
            lyrics, songID, songName = process_first_line(lyrics, songID, songName, row)
            songNumber = row['track_n']
            i += 1
    lyrics_data = pd.DataFrame({'songID': songID, 'songName': songName, 'lyrics': lyrics})
    training_number = round(i * 0.8)
    validation_number = i - training_number

    # Save Lyrics in .txt file
    with open('data/lyricsText.txt', 'w', encoding="utf-8") as filehandle:
        for listitem in lyrics:
            filehandle.write('%s\n' % listitem)
    with open('data/lyricsText_validation.txt', 'w', encoding="utf-8") as filehandle:
        for listitem in lyrics[training_number:]:
            filehandle.write('%s\n' % listitem)

file_path = "data/eminem_lyrics.txt"
raw_text = open(file_path, encoding='UTF-8').read()
raw_text = raw_text.lower()

chars = sorted(list(set(raw_text)))
int_chars = dict((i, c) for i, c in enumerate(chars))
chars_int = dict((i, c) for c, i in enumerate(chars))

n_chars = len(raw_text)
n_vocab = len(chars)

to_display_statistics = input("Display data statistics\n\t1-yes\n\tElse-no\n\t")
if to_display_statistics == "1":
    print(dataset.head())
    print(dataset.describe())
    print('Total Characters : ', n_chars)  # number of all the characters in lyricsText.txt
    print('Total Vocab : ', n_vocab)  # number of unique characters

train_model = input("\nTrain model\n\t1-yes\n\tElse-no\n\t")

model_name = "eminem_text_gen_rnn_model"

if train_model == "1":
    textgen = textgenrnn(name=model_name)
    train_function = textgen.train_from_file
    train_function(
        file_path=file_path,
        new_model=True,
        num_epochs=train_cfg['num_epochs'],
        gen_epochs=train_cfg['gen_epochs'],
        batch_size=train_cfg['batch_size'],
        train_size=train_cfg['train_size'],
        dropout=train_cfg['dropout'],
        max_gen_length=train_cfg['max_gen_length'],
        validation=train_cfg['validation'],
        is_csv=train_cfg['is_csv'],
        rnn_layers=model_cfg['rnn_layers'],
        rnn_bidirectional=model_cfg['rnn_bidirectional'],
        max_length=model_cfg['max_length'],
        dim_embeddings=model_cfg['dim_embeddings'],
        word_level=model_cfg['word_level'])
else:
    textgen = textgenrnn(name=model_name, weights_path="eminem_text_gen_rnn_model_weights.hdf5",
                         vocab_path="eminem_text_gen_rnn_model_vocab.json",
                         config_path="eminem_text_gen_rnn_model_config.json")

print(textgen.model.summary())

text = textgen.generate(20, temperature=1.0, return_as_list=True)
filename = input("\nEnter file name to save the song:\n")
filename = "results_textgenrnn/" + filename + ".txt"
text_file = open(filename, "w")
for line in text:
    print(line)
    text_file.write(line)
    text_file.write("\n")
text_file.close()
print("\nDone")
