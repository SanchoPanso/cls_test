import torch
from torchmetrics.classification import MultilabelAccuracy, MultilabelF1Score, BinaryAccuracy


def build_metrics(self):
    num_labels = len(self.num2label) 
    if num_labels == 1:
        self.accuracy1 = BinaryAccuracy()
        self.F1_M = lambda x, y: torch.tensor(0)
        self.F1_m = lambda x, y: torch.tensor(0)
        self.F1_N = lambda x, y: torch.tensor(0)
    else:
        self.accuracy1 = MultilabelAccuracy(
            num_labels=num_labels,
            average="micro",
        )
        self.F1_M = MultilabelF1Score(
            num_labels=num_labels,
            average="macro",
        )
        self.F1_m = MultilabelF1Score(
            num_labels=num_labels,
            average="micro",
        )
        self.F1_N = MultilabelF1Score(
            num_labels=num_labels,
            average="none",
        )


@torch.no_grad()
def get_metrics(self, logits, y):
    acc_m = self.accuracy1(logits, y)
    f1_m = self.F1_m(logits, y)
    f1_M = self.F1_M(logits, y)
    f1_N = self.F1_N(logits, y)
    return acc_m, f1_m, f1_M, f1_N

