import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist
import random

current_file = open("test.ft.txt", "rb")
x = current_file.read()
current_file.close()

x = x.decode("utf-8")
x = x.splitlines()
random.shuffle(x)
x = x[:10000]


labels = []
reviews = []
for i in x:
    separated = i.split(" ", 1)
    labels.append(separated[0])
    reviews.append(separated[1])

for i in range(len(labels)):
    if labels[i] == '__label__1':
        labels[i] = 'Bad'
    elif labels[i] == '__label__2':
        labels[i] = 'Good'
    else:
        print("Ruh-Roh")


reTokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()
all_words = []
for i in range(len(reviews)):
    tokens = reTokenizer.tokenize(reviews[i])         #### FIX IN GITHUB
    reviews[i] = []
    for word in tokens:
        word = word.lower()
        word = lemmatizer.lemmatize(word)
        all_words.append(word)
        reviews[i].append(word)

all_words = FreqDist(all_words)
most_common_words = all_words.most_common(3000)
word_features = []
for w in most_common_words:
    word_features.append(w[0])


def make_feature_dict(word_list):
    feature_dict = {}
    for w in word_features:
        if w in word_list:
            feature_dict[w] = 1
        else:
            feature_dict[w] = 0
    return feature_dict


nltk_data_set = []
for i in range(len(labels)):
    nltk_data_set.append( (make_feature_dict(reviews[i]), labels[i]) )

train_proportion = 0.8
train_set_size = int(train_proportion * len(labels))


accuracies = 0
for i in range(10):
    random.shuffle(nltk_data_set)

    training_set = nltk_data_set[:train_set_size]
    testing_set = nltk_data_set[train_set_size:]

    print("Beginning classifier training...")
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    print("Finished classifier training... its accuracy on the test set is: ")
    print(nltk.classify.accuracy(classifier, testing_set))
    accuracies += nltk.classify.accuracy(classifier, testing_set)

    #print("The most informative features were: ")
    #classifier.show_most_informative_features(30)


print("Average accuracy across 10 models: ")
print(accuracies/10)
