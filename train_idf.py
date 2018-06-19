from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


from sklearn.naive_bayes import MultinomialNB
import pickle

twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()
word_vectorizer = TfidfVectorizer(
    stop_words='english',
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{2,}',  #vectorize 2-character words or more
    ngram_range=(1, 1),
    max_features=30000)


word_vectorizer.fit(twenty_train.data)
X_train_word_features = word_vectorizer.transform(twenty_train.data)


text_clf = MultinomialNB().fit(X_train_word_features, twenty_train.target)


filename = 'text_clf.pkl'
pickle.dump(text_clf, open(filename, 'wb'))
pickle.dump(word_vectorizer, open('vectorizer.pkl', 'wb'))
