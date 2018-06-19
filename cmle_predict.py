from googleapiclient import discovery
from oauth2client.client import GoogleCredentials
import pickle
import numpy as np


from scipy.sparse import csr_matrix
from sklearn.datasets import fetch_20newsgroups

PROJECT_ID = "your-project-id"
VERSION_NAME = "v1"
MODEL_NAME = "text_clf"
BUCKET = "your-bucket"


credentials = GoogleCredentials.get_application_default()
api = discovery.build('ml', 'v1', credentials=credentials)

twenty_test = fetch_20newsgroups(subset='test', shuffle=True)

vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
test_features = vectorizer.transform(twenty_test)

dense_features = csr_matrix.todense(test_features)


input_data = {'instances': [np.squeeze(np.asarray(dense_features))[0].tolist()]}
parent = 'projects/%s/models/%s' % (PROJECT_ID, MODEL_NAME)
prediction = api.projects().predict(body=input_data, name=parent).execute()
print(prediction)
