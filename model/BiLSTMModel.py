from module.ESIM import *
from module.Utils import *
from module.CPUEmbedding import *
from transformers import BertModel

class BiLSTMModel(nn.Module):
    def __init__(self, vocab, config):
        super(BiLSTMModel, self).__init__()
        self.config = config
<<<<<<< HEAD
=======
        self.word_dims = 768 // 2
>>>>>>> bert

        self.rnn_dropout = RNNDropout(p=config.dropout_mlp)

        self.hidden_size = config.lstm_hiddens

        self.lstm_enc = Seq2SeqEncoder(nn.LSTM,
                                    2*self.word_dims,
                                    self.hidden_size,
                                    bidirectional=True)

        self.atten = SoftmaxAttention()

        self.hidden_dim = 4*2*config.lstm_hiddens

        self.mlp = nn.Sequential(nn.Linear(4*2*self.hidden_size,
                                        self.hidden_size),
                                        nn.ReLU())

        self.lstm_dec = Seq2SeqEncoder(nn.LSTM,
                                    self.hidden_size,
                                    self.hidden_size,
                                    bidirectional=True)

        self.feature_dim = 2*4*config.lstm_hiddens
        self.proj = nn.Sequential(nn.Dropout(p=config.dropout_mlp),
                                nn.Linear(2*4*self.hidden_size,  self.hidden_size),
                                nn.Tanh(),
                                nn.Dropout(p=config.dropout_mlp),
                                nn.Linear(self.hidden_size, vocab.tag_size))
        # 这步会初始化当前module里所有linear(包括bert的)和lstm，去掉，单独加初始化
        self.apply(_init_esim_weights)

<<<<<<< HEAD
=======
    def forward(self, src_embed, tgt_embed, src_lens, tgt_lens, src_masks, tgt_masks):
        ##unpack inputs

>>>>>>> bert
        src_embed = self.rnn_dropout(src_embed)
        tgt_embed = self.rnn_dropout(tgt_embed)

        src_hiddens = self.lstm_enc(src_embed, src_lens)
        tgt_hiddens = self.lstm_enc(tgt_embed, tgt_lens)


        src_hiddens_att, tgt_hiddens_att = self.atten(src_hiddens, src_masks, \
                                                      tgt_hiddens, tgt_masks)

        src_diff_hiddens = src_hiddens - src_hiddens_att
        src_prod_hiddens = src_hiddens * src_hiddens_att
        # [batch_size, src_len, 2 * lstm_hiddens * 4] 乘2是双向
        src_summary_hiddens = torch.cat([src_hiddens, src_hiddens_att, src_diff_hiddens, \
                                         src_prod_hiddens], dim=-1)

        tgt_diff_hiddens = tgt_hiddens - tgt_hiddens_att
        tgt_prod_hiddens = tgt_hiddens * tgt_hiddens_att
        tgt_summary_hiddens = torch.cat([tgt_hiddens, tgt_hiddens_att, tgt_diff_hiddens, \
                                         tgt_prod_hiddens], dim=-1)

        src_hiddens_proj = self.mlp(src_summary_hiddens)
        tgt_hiddens_proj = self.mlp(tgt_summary_hiddens)

        src_hiddens_proj = self.rnn_dropout(src_hiddens_proj)
        tgt_hiddens_proj = self.rnn_dropout(tgt_hiddens_proj)
        # [batch_size, src_len, 2 * lstm_hiddens]
        src_final_hiddens = self.lstm_dec(src_hiddens_proj, src_lens)
        tgt_final_hiddens = self.lstm_dec(tgt_hiddens_proj, tgt_lens)
        # 用一个vector表示src，先盖掉mask的state，再在length维度上求和，最后除以非mask个数，得到avgpool
        src_hidden_avg = torch.sum(src_final_hiddens * src_masks.unsqueeze(1)
                                                .transpose(2, 1), dim=1)\
            / (torch.sum(src_masks, dim=1, keepdim=True) + 1e-7)
        tgt_hidden_avg = torch.sum(tgt_final_hiddens * tgt_masks.unsqueeze(1)
                                                  .transpose(2, 1), dim=1)\
            / (torch.sum(tgt_masks, dim=1, keepdim=True) + 1e-7)
        # maxpool
        src_hidden_max, _ = replace_masked(src_final_hiddens, src_masks, -1e7).max(dim=1)
        tgt_hidden_max, _ = replace_masked(tgt_final_hiddens, tgt_masks, -1e7).max(dim=1)
        # [batch_size, 2 * lstm_hiddens * 4]
        hiddens = torch.cat([src_hidden_avg, src_hidden_max, tgt_hidden_avg, tgt_hidden_max], dim=1)
        # [batch_size, tag_size]
        outputs = self.proj(hiddens)
        return outputs


def _init_esim_weights(module):
    """
    Initialise the weights of the ESIM model.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)

    elif isinstance(module, nn.LSTM):
        nn.init.xavier_uniform_(module.weight_ih_l0.data)
        nn.init.orthogonal_(module.weight_hh_l0.data)
        nn.init.constant_(module.bias_ih_l0.data, 0.0)
        nn.init.constant_(module.bias_hh_l0.data, 0.0)
        hidden_size = module.bias_hh_l0.data.shape[0] // 4
        module.bias_hh_l0.data[hidden_size:(2*hidden_size)] = 1.0

        if (module.bidirectional):
            nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
            nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
            nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
            nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
            module.bias_hh_l0_reverse.data[hidden_size:(2*hidden_size)] = 1.0