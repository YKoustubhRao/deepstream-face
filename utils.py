from sklearn.preprocessing import Normalizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from datetime import datetime

def load_dataset():
	data = pd.read_csv("./data/embedding.csv", index_col=0)
	return data

def normalize_vectors(vectors):
	# normalize input vectors
	normalizer = Normalizer(norm='l2')
	vectors = normalizer.transform(vectors)
	return vectors


def predict_using_classifier(faces_embeddings, face_to_predict_embedding):
    return cosine_similarity(faces_embeddings, face_to_predict_embedding).reshape(-1)


def save_entry_log(id, cam_id):
    df = pd.DataFrame([[id, cam_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S")]])
    df.to_csv('./data/log.csv', mode='a', index=False, header=False)

def save_embeddings(vectors, label):
    df = pd.DataFrame(vectors, index=[label])
    df.to_csv('./data/embedding.csv', mode='a', header=False)