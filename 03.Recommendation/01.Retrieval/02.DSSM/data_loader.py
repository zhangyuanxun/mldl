import collections

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
from sklearn.model_selection import train_test_split
import random
from torch.utils.data import DataLoader, Dataset, TensorDataset

USER_FILE_PATH = "../../../00.Datasets/02.RS/01.MovieLen/ml-100k/u.user"
ITEM_FILE_PATH = "../../../00.Datasets/02.RS/01.MovieLen/ml-100k/u.item"
RATING_FILE_PATH_TRAIN = "../../../00.Datasets/02.RS/01.MovieLen/ml-100k/ua.base"
RATING_FILE_PATH_TEST = "../../../00.Datasets/02.RS/01.MovieLen/ml-100k/ua.test"

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


class MovieLensDataset:
    def __init__(self, train_rating_path, test_rating_path, user_path, item_path, min_threshold=4):
        self.user_cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
        self.item_cols = ['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url', 'unknown', 'action',
                          'adventure',
                          'animation', 'children', 'comedy', 'crime', 'documentary', 'drama', 'fantasy', 'film-Noir',
                          'horror',
                          'musical', 'mystery', 'romance', 'sci-fi', 'thriller', 'war', 'western']
        self.rating_cols = ['user_id', 'item_id', 'rating', 'timestamp']
        self.numerical_features = ['age', 'timestamp']
        self.categorical_features = ['user_id', 'item_id', 'gender', 'occupation', 'action', 'adventure', 'animation',
                                     'children', 'comedy', 'crime', 'documentary', 'drama', 'fantasy', 'film-Noir',
                                     'horror',
                                     'musical', 'mystery', 'romance', 'sci-fi', 'thriller', 'war', 'western']
        self.train_rating_path = train_rating_path
        self.test_rating_path = test_rating_path
        self.user_path = user_path
        self.item_path = item_path
        self.num_features = len(self.numerical_features) + len(self.categorical_features)
        self.field_dims = np.zeros(self.num_features, dtype=np.int64)

        self.item_df = pd.read_csv(ITEM_FILE_PATH, sep='|', names=self.item_cols)
        self.item_df = self.item_df.drop(columns=['title', 'release_date', 'video_release_date', 'imdb_url', 'unknown'])

        self.user_df = pd.read_csv(USER_FILE_PATH, sep='|', names=self.user_cols)
        self.user_df = self.user_df.drop(columns=['zip_code'])
        self.user_df[['age']] = np.round(MinMaxScaler().fit_transform(self.user_df[['age']]), 2)

        train_df = pd.read_csv(RATING_FILE_PATH_TRAIN, sep='\t', names=self.rating_cols)
        test_df = pd.read_csv(RATING_FILE_PATH_TEST, sep='\t', names=self.rating_cols)

        self.rating_df = pd.concat([train_df, test_df], axis=0)
        self.rating_df[['timestamp']] = np.round(MinMaxScaler().fit_transform(self.rating_df[['timestamp']]), 2)

        # positive sample (rating >=3), negative sample (rating < 3)
        self.rating_df['rating'] = self.rating_df.rating.apply(lambda x: 1 if int(x) >= 3 else 0)

        # merge with user and item
        self.rating_df = self.rating_df.merge(self.user_df, on='user_id', how='left')
        self.rating_df = self.rating_df.merge(self.item_df, on='item_id', how='left')

        self.features = self.rating_df.columns.values.tolist()
        self.features.remove('rating')
        self.feature2idx = {f: i for i, f in enumerate(self.features)}
        feature_cnts = defaultdict(lambda: defaultdict(int))
        for index, row in self.rating_df.iterrows():
            for feat in self.feature2idx.keys():
                feature_cnts[self.feature2idx[feat]][row[feat]] += 1

        self.feature_mapper = {i: {feat for feat, c in cnt.items() if
                                   c >= min_threshold or (i == self.feature2idx['user_id'] or i == self.feature2idx[
                                       "item_id"])} for i, cnt in feature_cnts.items()}
        self.feature_mapper = {i: {feat: idx for idx, feat in enumerate(cnt)} for i, cnt in self.feature_mapper.items()}
        self.defaults = {i: len(feat_pos) for i, feat_pos in self.feature_mapper.items()}
        for i, f in self.feature_mapper.items():
            self.field_dims[i] = len(f) + 1          # reserve one for unknown value
        self.offsets = np.array((0, *np.cumsum(self.field_dims)[:-1]))

    def load_data(self, negsample=2, test_ratio=0.1):
        item_ids = self.rating_df['item_id'].unique()
        train_set = []
        test_set = []

        for index, row in self.rating_df.groupby('user_id'):
            pos_list = row['item_id'].tolist()
            rating_list = row['rating'].tolist()

            user_feature = self.user_df[self.user_df['user_id'] == row['user_id'].tolist()[0]]
            user_feature = user_feature.values.tolist()[0]
            num_train = int(len(pos_list) * (1 - test_ratio))

            for i in range(num_train):
                item_feature = self.item_df[self.item_df['item_id'] == pos_list[i]]
                item_feature = item_feature.values.tolist()[0]

                train_set.append(user_feature + item_feature + [rating_list[i]])

            for i in range(num_train, len(pos_list)):
                item_feature = self.item_df[self.item_df['item_id'] == pos_list[i]]
                item_feature = item_feature.values.tolist()[0]

                test_set.append(user_feature + item_feature + [rating_list[i]])

            if negsample > 0:
                candidate_set = list(set(item_ids) - set(pos_list))
                neg_list = np.random.choice(candidate_set, size=len(pos_list) * negsample, replace=True)

                for i in range(len(neg_list)):
                    item_feature = self.item_df[self.item_df['item_id'] == neg_list[i]]
                    item_feature = item_feature.values.tolist()[0]

                    train_set.append(user_feature + item_feature + [0])

        random.shuffle(train_set)
        random.shuffle(test_set)

        feature_columns = self.user_df.columns.tolist() + self.item_df.columns.tolist() + ['label']

        train_set = pd.DataFrame(train_set, columns=feature_columns)
        test_set = pd.DataFrame(test_set, columns=feature_columns)
        self.user_feature_cols = list()
        self.item_feature_cols = list()
        for i, col in enumerate(self.user_df.columns.tolist()):
            if col == 'user_id':
                self.user_feature_cols.append({'feat': col, 'embed_dim': 32,
                                               'feat_num': self.defaults[self.feature2idx[col]] + 1})
            else:
                self.user_feature_cols.append({'feat': col, 'embed_dim': 8,
                                               'feat_num': self.defaults[self.feature2idx[col]] + 1})

        for i, col in enumerate(self.item_df.columns.tolist()):
            if col == 'item_id':
                self.item_feature_cols.append({'feat': col, 'embed_dim': 32,
                                               'feat_num': self.defaults[self.feature2idx[col]] + 1})
            else:
                self.item_feature_cols.append({'feat': col, 'embed_dim': 8,
                                               'feat_num': self.defaults[self.feature2idx[col]] + 1})

        return train_set, test_set

    def feature_transformation(self, data):
        feature_columns = list(self.feature2idx.keys())
        feature_columns.remove('timestamp')

        for col in feature_columns:
            mapper = self.feature_mapper[self.feature2idx[col]]
            data[col] = data[col].map(mapper)
            data[col] = data[col].fillna(self.defaults[self.feature2idx[col]])

        X, y = data.drop(columns='label').values, data['label'].values
        dataset = TensorDataset(torch.tensor(X).float(), torch.tensor(y).float())
        dataset = DataLoader(dataset, shuffle=True, batch_size=32)
        return dataset


if __name__ == "__main__":
    movielens_dataset = MovieLensDataset(train_rating_path=RATING_FILE_PATH_TRAIN,
                                         test_rating_path=RATING_FILE_PATH_TEST,
                                         user_path=USER_FILE_PATH,
                                         item_path=ITEM_FILE_PATH)

    train_set, test_set = movielens_dataset.load_data()
    train_dataset = movielens_dataset.feature_transformation(train_set)
    test_dataset = movielens_dataset.feature_transformation(test_set)


