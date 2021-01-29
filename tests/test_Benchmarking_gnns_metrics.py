import torch
import numpy as np
from sklearn.metrics import confusion_matrix

from topognn.metrics import WeightedAccuracy


def accuracy_SBM(scores, targets):
    S = targets.cpu().numpy()
    C = np.argmax(torch.nn.Softmax(dim=1)(
        scores).cpu().detach().numpy(), axis=1)
    CM = confusion_matrix(S, C).astype(np.float32)
    nb_classes = CM.shape[0]
    targets = targets.cpu().detach().numpy()
    nb_non_empty_classes = 0
    pr_classes = np.zeros(nb_classes)
    for r in range(nb_classes):
        cluster = np.where(targets == r)[0]
        if cluster.shape[0] != 0:
            pr_classes[r] = CM[r, r] / float(cluster.shape[0])
            if CM[r, r] > 0:
                nb_non_empty_classes += 1
        else:
            pr_classes[r] = 0.0
    acc = 100. * np.sum(pr_classes) / float(nb_classes)
    return acc


def test_weighted_accuracy():
    predictions = torch.tensor(np.random.rand(100, 10))
    labels = torch.tensor(np.random.randint(0, 10, size=100))
    gt_acc = accuracy_SBM(predictions, labels)
    acc = WeightedAccuracy(10)(predictions, labels)
    assert np.allclose(gt_acc, acc*100)


def test_chunked_computation():
    predictions = torch.tensor(np.random.rand(100, 10))
    labels = torch.tensor(np.random.randint(0, 10, size=100))
    acc_whole = WeightedAccuracy(10)
    acc_chunked = WeightedAccuracy(10)

    total_acc = acc_whole(predictions, labels)
    for p, l in zip(torch.split(predictions, 17), torch.split(labels, 17)):
        acc_chunked.update(p, l)

    assert total_acc == acc_chunked.compute()
