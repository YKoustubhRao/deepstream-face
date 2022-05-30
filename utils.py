from sklearn.preprocessing import Normalizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from datetime import datetime
import sqlite3

conn = sqlite3.connect('./data/db.sqlite')

def load_dataset():
	df = pd.read_sql('select * from embeddings', conn)
	names = df[df.columns[0:1]]
	embeddings = df[df.columns[1:385]]
	return names.to_numpy(), embeddings.to_numpy(dtype=np.float64)

def normalize_vectors(vectors):
	# normalize input vectors
	normalizer = Normalizer(norm='l2')
	vectors = normalizer.transform(vectors)
	return vectors


def predict_using_classifier(faces_embeddings, face_to_predict_embedding):
    return cosine_similarity(faces_embeddings, face_to_predict_embedding).reshape(-1)


def save_entry_log(id, cam_id):
	df = pd.DataFrame([[id, cam_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S")]], columns=["name", "timestamp"])
	df.to_sql('logs', conn, if_exists='append', index=False)

def save_embeddings(vectors, label):
    df = pd.DataFrame(np.array([label]) + vectors, columns=["name"] + list(range(0, 384)))
    df.to_sql('embeddings', conn, if_exists='append', index=False)