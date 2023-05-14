import torch
from torchmetrics.functional.classification import auroc

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def roc_auc(output, target):
    if len(target.shape) == 1:
        target = target.unsqueeze(-1)

    rocauc_list = auroc(output, target.type(torch.long), num_classes=target.shape[1], average=None)
    return rocauc_list.mean()