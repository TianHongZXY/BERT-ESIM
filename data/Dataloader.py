from data.Vocab import *
from data.Instance import *
from data.TensorInstances import *
from collections import Counter
import codecs
import pysnooper
from tqdm import tqdm
def read_corpus(file, tokenizer):
    '''读取语料，一条语料可能有多句，不同条语料之间有一个空行'''
    data = []
    with codecs.open(file, encoding='utf8') as input_file:
        curtext = []
        for line in tqdm(input_file.readlines()):
            line = line.strip()
            if line is not None and line != '': # 非空行都是一条数据
                curtext.append(line)
            else:
                slen = len(curtext) # 读到空行时 curtext包含 src tgt 和 label
                if slen == 3:
                    cur_data = parseInstance(curtext, tokenizer)
                    if cur_data.src_len <= 500 and cur_data.tgt_len <=500:
                        data.append(cur_data)
                curtext = []
                # TODO debug使用
                # if(len(data)) >= 10:
                #     break

    slen = len(curtext)
    if slen == 3:
        cur_data = parseInstance(curtext)
        if cur_data.src_len <= 500 and cur_data.tgt_len <= 500:
            data.append(cur_data)

    print("Total num: " + str(len(data)))
    return data

def creatVocab(corpusFile, min_occur_count, tokenizer):
    '''根据语料创建Vocab'''
    word_counter = Counter()
    tag_counter = Counter()
    alldatas = read_corpus(corpusFile, tokenizer)
    for inst in alldatas:
        # for curword in inst.src_forms:
        #     word_counter[curword] += 1
        # for curword in inst.tgt_forms:
        #     word_counter[curword] += 1
        tag_counter[inst.tag] += 1

    return Vocab(word_counter, tag_counter, min_occur_count), alldatas

def insts_numberize(insts, vocab, tokenizer):
    for inst in insts:
        yield inst2id(inst, vocab, tokenizer)

def inst2id(inst, vocab, tokenizer):
    tagid = vocab.tag2id(inst.tag)
    return (inst.src_bert_indice, inst.src_segments_id, inst.src_piece_id), \
           (inst.tgt_bert_indice, inst.tgt_segments_id, inst.tgt_piece_id), tagid


def batch_slice(data, batch_size):
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        insts = [data[i * batch_size + b] for b in range(cur_batch_size)]

        yield insts


def data_iter(data, batch_size, shuffle=True):
    """
    randomly permute data, then sort by source length, and partition into batches
    ensure that the length of  insts in each batch
    """

    batched_data = []
    if shuffle: np.random.shuffle(data)
    batched_data.extend(list(batch_slice(data, batch_size)))

    if shuffle: np.random.shuffle(batched_data)
    for batch in batched_data:
        yield batch

# @pysnooper.snoop()
def batch_data_variable(batch, vocab, tokenizer):
    slen, tlen = batch[0].src_len, batch[0].tgt_len
    batch_size = len(batch)
    for b in range(1, batch_size):
        cur_slen, cur_tlen = batch[b].src_len, batch[b].tgt_len
        if cur_slen > slen: slen = cur_slen
        if cur_tlen > tlen: tlen = cur_tlen
    tinst = TensorInstances(batch_size, slen, tlen)

    b = 0
    for (src_bert_indice, src_segments_id, src_piece_id), (tgt_bert_indice, tgt_segments_id, tgt_piece_id), tagid in insts_numberize(batch, vocab, tokenizer):
        tinst.tags[b] = tagid
        cur_slen, cur_tlen = len(src_bert_indice), len(tgt_bert_indice)
        tinst.src_lens[b] = cur_slen
        tinst.tgt_lens[b] = cur_tlen
        # print(src_bert_indice)
        # print('curslen', cur_slen)
        # print(tinst.src_bert_indice.size())
        tinst.src_bert_indice[b][:cur_slen] = torch.LongTensor(np.array(src_bert_indice)) # list -> array -> tensor
        tinst.tgt_bert_indice[b][:cur_tlen] = torch.LongTensor(np.array(tgt_bert_indice)) # 直接 list -> tensor 会内存泄漏
        tinst.src_masks[b][:cur_slen] = 1
        tinst.tgt_masks[b][:cur_tlen] = 1

        b += 1
    return tinst

def batch_variable_inst(insts, tagids, vocab, tokenizer):
    for inst, tagid in zip(insts, tagids):
        pred_tag = vocab.id2tag(tagid)
        yield Instance(inst.src_words, inst.tgt_words, pred_tag, tokenizer), pred_tag == inst.tag
