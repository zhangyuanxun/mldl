import torch.optim

from model import DSSM
from data_loader import *
from sklearn.metrics import roc_auc_score
import datetime
import torch.nn as nn
from torchkeras import summary, Model


def auc(y_pred, y_true):
    pred = y_pred.data
    y = y_true.data
    return roc_auc_score(y, pred)


def main():
    movielens_dataset = MovieLensDataset(train_rating_path=RATING_FILE_PATH_TRAIN,
                                         test_rating_path=RATING_FILE_PATH_TEST,
                                         user_path=USER_FILE_PATH,
                                         item_path=ITEM_FILE_PATH)

    train_dataset, test_dataset = movielens_dataset.load_data()
    train_dataset, xxxx = movielens_dataset.feature_transformation(train_dataset)
    test_dataset, xxxx = movielens_dataset.feature_transformation(test_dataset)

    model = DSSM(user_feature_columns=movielens_dataset.user_feature_cols,
                 item_feature_columns=movielens_dataset.item_feature_cols)

    summary(model, input_shape=(xxxx,))

    loss_func = nn.BCELoss()
    print(list(model.parameters()))
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    metric_func = auc
    metric_name = 'auc'
    epochs = 4
    log_step_freq = 1000

    print('start_training.........')
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('========' * 8 + '%s' % nowtime)

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        metric_sum = 0.0
        step = 1

        for step, (features, labels) in enumerate(train_dataset, 1):
            optimizer.zero_grad()

            predictions = model(features)
            loss = loss_func(predictions, labels)
            try:
                metric = metric_func(predictions, labels)
            except ValueError:
                pass

            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            metric_sum += metric.item()
            if step % log_step_freq == 0:
                print(("[step=%d] loss: %.3f, " + metric_name + ": %.3f") % (step, loss_sum / step, metric_sum / step));

        info = (epoch, loss_sum / step, metric_sum / step)
        print(("\nEPOCH=%d, loss=%.3f, " + metric_name + " = %.3f") % info)
        nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print('\n' + '==========' * 8 + '%s' % nowtime)

    all_item_input = movielens_dataset.item_profile.values
    all_item_input = DataLoader(torch.tensor(all_item_input).float(), shuffle=True, batch_size=128)
    item_embeddings = torch.tensor([], device='cpu')
    model.eval()
    for batch_idx, item_input in enumerate(all_item_input):
        item_embed = model.generate_item_embedding(item_input)
        item_embeddings = torch.cat((item_embeddings, item_embed), 0)



if __name__ == "__main__":
    main()