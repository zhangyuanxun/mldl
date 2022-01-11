import numpy as np
import faiss
from tqdm import tqdm
from deepmatch.utils import recall_N
import pandas as pd
import pickle

with open('idx2itemid.pkl', 'rb') as f:
    idx2itemid = pickle.load(f)

print(idx2itemid.keys())
with open('idx2userid.pkl', 'rb') as f:
    idx2userid = pickle.load(f)

item_embedding_matrix = np.load("item_embedding_matrix.npy")
user_embedding_matrix = np.load("user_embedding_matrix.npy")

index = faiss.IndexFlatIP(item_embedding_matrix.shape[1])
# faiss.normalize_L2(item_embs)
index.add(item_embedding_matrix)
# faiss.normalize_L2(user_embs)

D, I = index.search(user_embedding_matrix, 50)
u2i_reco = list()
for i in range(I.shape[0]):
    user_id = idx2userid[i]
    item_ids_lst = I[i].tolist()
    item_ids_lst = [idx2itemid[_id] for _id in item_ids_lst]
    arr = [user_id] + item_ids_lst
    u2i_reco.append(arr)

cols = []
for i in range(50):
    cols.append("Top " + str(i + 1))
cols = ['user id'] + cols

u2i_reco = pd.DataFrame(u2i_reco, columns=cols)
test_dataset = pd.read_csv("movielens_test.csv")
test_user_item_ids = test_dataset[['user_id', 'item_id']]

recall = list()
for idx, row in test_user_item_ids.groupby('user_id'):
    user_id = row['user_id'].tolist()[0]

    true_product_ids = row['item_id'].tolist()
    pred_product_ids = u2i_reco[u2i_reco['user id'] == user_id]
    pred_product_ids = pred_product_ids.values.tolist()[0][1:]
    recall.append(recall_N(true_product_ids, pred_product_ids, N=50))

print("recall at 50: {}".format(np.mean(recall)))
