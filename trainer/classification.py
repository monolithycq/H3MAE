import torch
from tqdm import tqdm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def get_rep_with_label(model, dataloader,indexes,args):
    results = []
    labels = []
    with torch.no_grad():
        tqdm_train_dataloader = tqdm(dataloader)
        for i, data in enumerate(tqdm_train_dataloader):

            input, label = map(lambda x: x.to(args.device), data)

            labels += label.cpu().numpy().tolist()
            result = model(input.float(),indexes)
            results += result.cpu().numpy().tolist()
    return results, labels

def get_predcls_with_label(model, dataloader, indexes, args):
    predicts = []
    labels = []
    tqdm_dataloader = tqdm(dataloader)
    with torch.no_grad():
        for i, data in enumerate(tqdm_dataloader):
            input, label = map(lambda x: x.to(args.device), data)
            output = model(input.float(), indexes)
            _, pred = torch.topk(output, 1)
            labels += label.cpu().numpy().tolist()
            predicts += pred.view(-1).cpu().numpy().tolist()
    return predicts,labels



def fit_lr(features, y, seed):
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            random_state=seed,
            max_iter=100,
            multi_class='ovr'
        )
    )
    pipe.fit(features, y)
    return pipe