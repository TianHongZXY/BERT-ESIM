import torch.nn.functional as F
import torch.nn as nn
from transformers import BertModel


class BiSententClassifier(object):
    def __init__(self, model, vocab):
        self.model = model
        p = next(filter(lambda p: p.requires_grad, model.parameters()))
        self.use_cuda = p.is_cuda
        self.device = p.get_device() if self.use_cuda else None
        self.criterion = nn.CrossEntropyLoss()
        pretrained_weights = '/Users/tianhongzxy/.cache/torch/transformers/bert-base-uncased'
        self.bert = BertModel.from_pretrained(pretrained_weights,
                                          # output_hidden_states=True,
                                          # output_attentions=True)
                                         )
        for p in self.bert.named_parameters():
            p[1].requires_grad = False

    def forward(self, tinputs):
        src_bert_indice, src_segments_id, src_piece_id, src_lens, src_masks, \
        tgt_bert_indice, tgt_segments_id, tgt_piece_id, tgt_lens, tgt_masks = tinputs

        src_embed, src_attn = self.bert(input_ids=src_bert_indice,
                        attention_mask=src_masks,
                        token_type_ids=src_segments_id,
                        # position_ids=src_piece_id,
                        )
        tgt_embed, tgt_attn = self.bert(input_ids=tgt_bert_indice,
                        attention_mask=tgt_masks,
                        token_type_ids=tgt_segments_id,
                        # position_ids=tgt_piece_id,
                        )
        tag_logits = self.model(src_embed, tgt_embed, src_lens, tgt_lens, src_masks, tgt_masks)
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
