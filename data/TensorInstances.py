import torch


class TensorInstances:
    def __init__(self, batch_size, slen, tlen):
        self.tags = torch.zeros(size=(batch_size, ), requires_grad=False, dtype=torch.long)

        self.src_masks = torch.zeros(size=(batch_size, slen), requires_grad=False, dtype=torch.long)
        self.src_lens = torch.zeros(size=(batch_size, ), requires_grad=False, dtype=torch.long)
        self.src_bert_indice =  torch.zeros(size=(batch_size, slen), requires_grad=False, dtype=torch.long)
        self.src_segments_id =  torch.zeros(size=(batch_size, slen), requires_grad=False, dtype=torch.long)
        self.src_piece_id = torch.arange(start=0, end=slen, requires_grad=False, dtype=torch.long)

        self.tgt_masks = torch.zeros(size=(batch_size, tlen), requires_grad=False, dtype=torch.long)
        self.tgt_lens = torch.zeros(size=(batch_size, ), requires_grad=False, dtype=torch.long)
        self.tgt_bert_indice = torch.zeros(size=(batch_size, tlen), requires_grad=False, dtype=torch.long)
        self.tgt_segments_id = torch.zeros(size=(batch_size, tlen), requires_grad=False, dtype=torch.long)
        self.tgt_piece_id = torch.arange(start=0, end=tlen, requires_grad=False, dtype=torch.long)

    def to_cuda(self, device):

        self.src_lens = self.src_lens.cuda(device)
        self.src_masks = self.src_masks.cuda(device)
        self.tgt_lens = self.tgt_lens.cuda(device)
        self.tgt_masks = self.tgt_masks.cuda(device)
        self.tags = self.tags.cuda(device)

        self.src_bert_indice = self.src_bert_indice.cuda(device)
        self.src_segments_id = self.src_segments_id.cuda(device)
        self.src_piece_id = self.src_piece_id.cuda(device)

        self.tgt_bert_indice = self.tgt_bert_indice.cuda(device)
        self.tgt_segments_id = self.tgt_segments_id.cuda(device)
        self.tgt_piece_id = self.tgt_piece_id.cuda(device)

    @property
    def inputs(self):
        return (self.src_bert_indice, self.src_segments_id, self.src_piece_id, self.src_lens, self.src_masks,\
                self.tgt_bert_indice, self.tgt_segments_id, self.tgt_piece_id, self.tgt_lens, self.tgt_masks)
    @property
    def outputs(self):
        return self.tags
