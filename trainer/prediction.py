import torch
from tqdm import tqdm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,LinearRegression


def get_pred_with_label(model, dataloader,indexes,args):
    results = []
    labels = []
    with torch.no_grad():
        tqdm_train_dataloader = tqdm(dataloader)
        for i, data in enumerate(tqdm_train_dataloader):
            input, label = map(lambda x: x.to(args.device), data)
            result = model.predict(input.float(), indexes)

            labels += label.squeeze().cpu().numpy().tolist()
            results += result.cpu().numpy().tolist()
    return results, labels

def fit_lr_pred(features, y):
    pipe = make_pipeline(
        StandardScaler(),
        LinearRegression()
    )
    pipe.fit(features, y)
    return pipe