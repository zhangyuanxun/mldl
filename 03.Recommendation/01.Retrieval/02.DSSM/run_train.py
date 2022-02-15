import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch
from model import DSSM, hinge_loss
from sklearn.metrics import roc_auc_score, accuracy_score
import datetime
import torch.nn as nn
import pickle
from feature_columns import SparseFeat, DenseFeat, SeqSparseFeat

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

RATING_FILE_PATH_TRAIN = "../../../00.Datasets/02.RS/01.MovieLen/sample/movielens_sample.csv"


def auc(y_pred, y_true):
    pred = y_pred.data.cpu()
    y = y_true.data.cpu()
    return roc_auc_score(y, pred)


def acc(y_pred, y_true):
    pred = y_pred.data.cpu()
    pred = (pred > 0.5)
    y = y_true.data.cpu()
    return accuracy_score(y, pred)


def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    num_samples = len(sequences)

    lengths = []
    sample_shape = ()
    flag = True

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.

    for x in sequences:
        try:
            lengths.append(len(x))
            if flag and len(x):
                sample_shape = np.asarray(x).shape[1:]
                flag = False
        except TypeError:
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))

    if maxlen is None:
        maxlen = np.max(lengths)

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, str) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def generate_train_test_dataset(data, neg_sample=3, test_ratio=0.1):
    data.sort_values("timestamp", inplace=True)
    item_ids = data['movie_id'].unique()

    train_set = []
    test_set = []
    for user_id, row in tqdm(data.groupby('user_id')):
        pos_list = row['movie_id'].tolist()
        rating_list = row['rating'].tolist()
        rating_list = [1 if rating > 3 else 0 for rating in rating_list]
        pos_list = [k for k, v in zip(pos_list, rating_list) if v > 0]
        if neg_sample > 0:
            candidate_set = list(set(item_ids) - set(pos_list))
            neg_list = np.random.choice(candidate_set, size=len(pos_list) * neg_sample, replace=True)
        for i in range(1, len(pos_list)):
            hist = pos_list[:i]

            if i != len(pos_list) - 1:
                train_set.append([user_id, pos_list[i], hist[::-1], len(hist[::-1]), rating_list[i]])

                for j in range(neg_sample):
                    train_set.append([user_id, neg_list[i * neg_sample + j],  hist[::-1], len(hist[::-1]), 0])
            else:
                test_set.append([user_id, pos_list[i], hist[::-1], len(hist[::-1]), rating_list[i]])

    random.shuffle(train_set)
    random.shuffle(test_set)
    print(len(train_set), len(test_set))

    return train_set, test_set


def generate_feature(dataset, user_profile, item_profile, batch_size, max_seq_len):
    user_id = np.array([line[0] for line in dataset])            # user_id
    item_id = np.array([line[1] for line in dataset])            # item_id
    his_seq = [line[2] for line in dataset]                      # history sequence item id
    hist_seq_len = np.array([line[3] for line in dataset])       # history sequence item length
    label = np.array([line[4] for line in dataset])              # label
    gender = user_profile.loc[user_id]['gender'].values          # gender
    age = user_profile.loc[user_id]['age'].values                # age
    occupation = user_profile.loc[user_id]['occupation'].values  # occupation
    zip = user_profile.loc[user_id]['zip'].values                # zip

    assert len(user_id) == len(item_id) == len(label) == len(gender) \
           == len(occupation) == len(zip) == len(age)

    his_seq_padded = pad_sequences(his_seq, maxlen=max_seq_len, padding='post', truncating='post', value=0)

    # Note, put user features before item feature
    X = np.stack((user_id, gender, occupation, zip, age), axis=1)
    X = np.concatenate((X, his_seq_padded, item_id.reshape(-1, 1)), axis=1)
    y = label

    dataset = TensorDataset(torch.tensor(X).float(), torch.tensor(y).float())
    dataset = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    return dataset


def main():
    data = pd.read_csv(RATING_FILE_PATH_TRAIN)
    neg_sample = 3
    batch_size = 128
    max_seq_len = 50
    sparse_features = ['user_id', 'movie_id', 'gender', 'occupation', 'zip']
    dense_features = ['age']
    print(data.head(10))

    feature_max_id = {}
    for feature in sparse_features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature]) + 1
        feature_max_id[feature] = data[feature].max() + 1

    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # define features
    user_sparse_features = ["user_id", "gender", "occupation", "zip"]
    user_dense_features = ["age"]

    item_sparse_features = ["movie_id"]

    user_profile = data[user_sparse_features + user_dense_features].drop_duplicates('user_id')
    item_profile = data[item_sparse_features].drop_duplicates('movie_id')
    user_profile.set_index("user_id", drop=False, inplace=True)

    print("Generate train and test dataset...")
    train_set, test_set = generate_train_test_dataset(data, neg_sample)

    print("Generate train and test features...")
    train_dataloader = generate_feature(train_set, user_profile, item_profile, batch_size, max_seq_len)
    test_dataloader = generate_feature(test_set, user_profile, item_profile, batch_size, max_seq_len)

    print("Generate feature columns...")
    embedding_dim = 8
    user_feature_columns = [SparseFeat(feat, feature_max_id[feat], embedding_dim) for i, feat in enumerate(user_sparse_features)] \
        + [DenseFeat(feat, 1) for i, feat in enumerate(user_dense_features)] \
        + [SeqSparseFeat(SparseFeat('user_hist', feature_max_id['movie_id'], embedding_dim, embedding_name='movie_id'),
                         maxlen=max_seq_len,combiner='mean', length_name=None)]

    item_feature_columns = [SparseFeat(feat, feature_max_id[feat], embedding_dim) for i, feat in enumerate(item_sparse_features)]

    # define model
    model = DSSM(user_feature_columns=user_feature_columns,
                 item_feature_columns=item_feature_columns)

    loss_func = nn.BCELoss()
    optimizer = torch.optim.Adagrad(params=model.parameters(), lr=0.2)
    metric_func = auc
    metric_name = 'auc'
    epochs = 3
    log_step_freq = 1000

    print('start_training.........')
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('========' * 8 + '%s' % nowtime)

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        metric_sum = 0.0
        step = 1

        for step, (features, labels) in enumerate(train_dataloader, 1):
            optimizer.zero_grad()

            out_pred, out_logits, cos_sim = model(features)
            loss = loss_func(out_pred, labels)
            try:
                metric = metric_func(out_pred, labels)
            except ValueError:
                pass

            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            metric_sum += metric.item()
            if step % log_step_freq == 0:
                print(("[step=%d] loss: %.3f, " + metric_name + ": %.3f") % (
                step, loss_sum / step, metric_sum / step));

        model.eval()
        val_loss_sum = 0.0
        val_metric_sum = 0.0

        for val_step, (features, labels) in enumerate(test_dataloader, 1):
            with torch.no_grad():
                out_pred, _, cos_sim = model(features)
                val_loss = loss_func(out_pred, labels)
                val_metric = acc(out_pred, labels)
            val_loss_sum += val_loss.item()
            val_metric_sum += val_metric.item()

        info = (epoch, loss_sum / step, metric_sum / step)
        print(("\nEPOCH=%d, val_loss=%.3f, " + "val_acc" + " = %.3f") % info)
        nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print('\n' + '==========' * 8 + '%s' % nowtime)

    item_inputs = item_profile['movie_id'].values.reshape(-1, 1)
    user_inputs = user_profile.values

    item_inputs_loader = DataLoader(torch.tensor(item_inputs).float(), batch_size=batch_size*4)
    user_inputs_loader = DataLoader(torch.tensor(user_inputs).float(), batch_size=batch_size*4)

    item_embeddings = torch.tensor([], device='cpu')
    user_embeddings = torch.tensor([], device='cpu')

    model.eval()
    for batch_idx, item_input in enumerate(item_inputs_loader):
        item_embed = model.generate_item_embedding(item_input)
        item_embeddings = torch.cat((item_embeddings, item_embed), 0)

    for batch_idx, user_input in enumerate(user_inputs_loader):
        user_embed = model.generate_user_embedding(user_input)
        user_embeddings = torch.cat((user_embeddings, user_embed), 0)

    print(item_embeddings.shape)
    print(user_embeddings.shape)

    item_embedding_matrix = item_embeddings.data.cpu().numpy()
    user_embedding_matrix = user_embeddings.data.cpu().numpy()

    np.save("item_embedding_matrix.npy", item_embedding_matrix)
    np.save("user_embedding_matrix.npy", user_embedding_matrix)

    item_profile.reset_index(inplace=True, drop=True)
    user_profile.reset_index(inplace=True, drop=True)
    idx2userid = dict()
    idx2itemid = dict()

    for idx, row in item_profile.iterrows():
        idx2itemid[idx] = row['movie_id']

    for idx, row in user_profile.iterrows():
        idx2userid[idx] = row['user_id']

    with open("idx2userid.pkl", 'wb') as fp:
        pickle.dump(idx2userid, fp)

    with open("idx2itemid.pkl", 'wb') as fp:
        pickle.dump(idx2itemid, fp)


if __name__ == "__main__":
    main()
