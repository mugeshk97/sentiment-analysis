import re
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
stemmer=WordNetLemmatizer()

model = tf.keras.models.load_model('senti.h5')

def sentiment(sentence):
    review = re.sub('[^a-zA-Z]', ' ',sentence)
    review = review.lower()
    review = review.split()
    review = [stemmer.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    ohe=tf.keras.preprocessing.text.one_hot(review,5000) 
    ip = tf.keras.preprocessing.sequence.pad_sequences([ohe], maxlen=500)
    prob = model.predict(ip)[0][0]
    return prob

