import torch.nn.functional as F
import torch.nn as nn


class BiSententClassifier(object):
    def __init__(self, model, vocab):
        self.model = model
        p = next(filter(lambda p: p.requires_grad, model.parameters()))
        self.use_cuda = p.is_cuda
        self.device = p.get_device() if self.use_cuda else None
        self.criterion = nn.CrossEntropyLoss()


    def forward(self, tinputs):
        tag_logits = self.model(tinputs)
        # cache
        self.tag_logits = tag_logits

    def compute_loss(self, true_tags):
        loss = self.criterion(self.tag_logits, true_tags)
        return loss

    def compute_accuracy(self, true_tags):
        b, l = self.tag_logits.size()
        # max(1)返回每行在列上的最大值和索引, [1]是取索引，[0]是取值
        pred_tags = self.tag_logits.detach().max(1)[1].cpu()
        true_tags = true_tags.detach().cpu()
        tag_correct = pred_tags.eq(true_tags).cpu().sum()

        return tag_correct, b

    def classifier(self, tinputs):
        if tinputs[0] is not None:
            self.forward(tinputs)
        pred_tags = self.tag_logits.detach().max(1)[1].cpu()
        return pred_tags
