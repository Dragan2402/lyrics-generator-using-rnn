import nltk
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import numpy as np
import seaborn

from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.optimizer_v2.adamax import Adamax
from PIL import Image, ImageDraw, ImageFont
from keras.utils import np_utils
from wordcloud import WordCloud, STOPWORDS

warnings.filterwarnings("ignore")
data = pd.read_csv("data/TaylorSwift.csv")
model = Sequential()


def data_display():
    global data
    data = data.iloc[:20, :]
    print("Artists in the data:\n", data.Artist.value_counts())
    print("Size of Dataset represented in total number of songs:", data.shape)

    data["No_of_Characters"] = data["Lyrics"].apply(len)
    data["No_of_Words"] = data.apply(lambda row: nltk.word_tokenize(row["Lyrics"]), axis=1).apply(len)
    data["No_of_Lines"] = data["Lyrics"].str.split('\n').apply(len)
    print(data.describe())
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(stopwords=stopwords, background_color="#444160", colormap="Purples", max_words=800).generate(
        " ".join(data["Lyrics"]))
    plt.figure(figsize=(12, 12))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.show()


def generate_corpus():
    corpus = ''
    for list_item in data.Lyrics:
        corpus += list_item

    corpus = corpus.lower()  # converting all alphabets to lowecase
    print("Number of unique characters before filtration:", len(set(corpus)))
    print("The unique characters before filtration:", sorted(set(corpus)))
    to_remove = ['{', '}', '~', '©', 'à', 'á', 'ã', 'ä', 'ç', 'è', 'é', 'ê', 'ë', 'í', 'ñ', 'ó', 'ö', 'ü', 'ŏ',
                 'е', 'ا', 'س', 'ل', 'م', 'و', '\u2005', '\u200a', '\u200b', '–', '—', '‘', '’', '‚', '“', '”',
                 '…', '\u205f', '\ufeff', '!', '&', '(', ')', '*', '-', '/', ]
    for symbol in to_remove:
        corpus = corpus.replace(symbol, " ")
    print("Number of unique characters after filtration :", len(set(corpus)))
    print("The unique characters after filtration :", sorted(set(corpus)))

    symbol = sorted(list(set(corpus)))

    l_corpus = len(corpus)  # length of corpus
    l_symbol = len(symbol)  # length of total unique characters

    # Building dictionary to access the vocabulary from indices and vice versa
    mapping = dict((c, i) for i, c in enumerate(symbol))
    reverse_mapping = dict((i, c) for i, c in enumerate(symbol))

    print("Total number of characters:", l_corpus)
    print("Number of unique characters:", l_symbol)

    # Splitting the Corpus in equal length of strings and output target
    length = 40
    features = []
    targets = []
    for i in range(0, l_corpus - length, 1):
        feature = corpus[i:i + length]
        target = corpus[i + length]
        features.append([mapping[j] for j in feature])
        targets.append(mapping[target])

    l_datapoints = len(targets)
    print("Total number of sequences in the Corpus:", l_datapoints)

    x = (np.reshape(features, (l_datapoints, length, 1))) / float(l_symbol)

    # one hot encode the output variable
    y = np_utils.to_categorical(targets)

    return x, y, l_symbol, mapping, reverse_mapping,corpus


def create_model(x, y):
    # Adding layers
    model.add(LSTM(256, input_shape=(x.shape[1], x.shape[2])))
    model.add(Dense(y.shape[1], activation='softmax'))
    # Compiling the model for training
    opt = Adamax(learning_rate=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=opt)

    # Model's Summary
    model.summary()


def train_model(x, y):
    history = model.fit(x, y, batch_size=64, epochs=5)
    history_df = pd.DataFrame(history.history)
    # Plotting the learnings

    fig = plt.figure(figsize=(15, 4), facecolor="#B291B6")
    fig.suptitle("Learning Plot of Model for Loss")
    pl = seaborn.lineplot(data=history_df["loss"], color="#444160")
    pl.set(ylabel="Training Loss")
    pl.set(xlabel="Epochs")







# The function to generate text from model
def lyrics_generator(starter, ch_count, l_symbol, mapping, reverse_mapping):  # ,temperature=1.0):
    generated = ""
    starter = starter
    seed = [mapping[char] for char in starter]
    generated += starter
    # Generating new text of given length
    for i in range(ch_count):
        seed = [mapping[char] for char in starter]
        x_pred = np.reshape(seed, (1, len(seed), 1))
        x_pred = x_pred / float(l_symbol)
        prediction = model.predict(x_pred, verbose=0)[0]
        # Getting the index of the next most probable index
        prediction = np.asarray(prediction).astype('float64')
        prediction = np.log(prediction) / 1.0
        exp_preds = np.exp(prediction)
        prediction = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, prediction, 1)
        index = np.argmax(prediction)
        next_char = reverse_mapping[index]
        # Generating new text
        generated += next_char
        starter = starter[1:] + next_char

    return generated


def main():
    global data
    print("Hello to the lyrics generator !!!\n\n")
    data_display()
    x, y, l_symbol, mapping, reverse_mapping ,corpus= generate_corpus()

    create_model(x, y)
    train_model(x, y)

    song_2 = lyrics_generator("i'm a sunflower, a little funny and cute", 400, l_symbol, mapping, reverse_mapping)
    for index in song_2:
        print(reverse_mapping[int(index)])


if __name__ == "__main__":
    main()
