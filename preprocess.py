#import statements
import csv
import glob
import pandas as pd
import re, string, unicodedata
import nltk
import inflect
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import nltk.classify.util
from nltk.stem import LancasterStemmer, WordNetLemmatizer

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def rem_punctuation(text):
    punc_removed = text.translate(str.maketrans('', '', string.punctuation))
    return punc_removed

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    #words = replace_numbers(words)
    words = remove_stopwords(words)
    return words

def stem_and_lemmatize(words):
    stems = stem_words(words)
    lemmas = lemmatize_verbs(words)
    return stems, lemmas


def preprocess():
    datafile = "C:/Users/Dante/PycharmProjects/DepressionAnalysis/datasets/classifier1_datasetA_Combined.csv"
    tweets = pd.read_csv(datafile, encoding='latin1')
    #texts = texts[:200]
    #labels = texts.iloc[:,0]
    #raw_text = texts.iloc[:,1]
    #print(raw_text)

    outfile = "C:/Users/Dante/PycharmProjects/DepressionAnalysis/datasets/processed_classifier.csv"
    f_out = open(outfile, mode='w+', newline='')
    write = csv.writer(f_out, quotechar='"')
    write.writerow(['message', 'label'])
    for index, row in tweets.iterrows():
        text = row[1]
        label = row[0]
        if isinstance(text, float):
            continue
        #html_free = strip_html(text)
        url_free = re.sub('http[s]?://\S+', '', text)
        number_free = re.sub('[0-9]+', '', url_free)
        words = nltk.word_tokenize(number_free)
        words = normalize(words)
        stems, lemmas = stem_and_lemmatize(words)
        normalized = " ".join(lemmas)
        if(normalized != ''):
            write.writerow(['{}'.format(normalized), '{}'.format(label)])

preprocess()
