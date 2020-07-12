class Instance:
    def __init__(self, src_words, tgt_words, tag, tokenizer):
        self.src_words = src_words
        self.src_forms = [curword for curword in src_words]
        self.tgt_words = tgt_words
        self.tgt_forms = [curword for curword in tgt_words]
        self.tag = tag
        self.tokenizer = tokenizer
        self.src_bert_indice, self.src_segments_id, self.src_piece_id = self.tokenizer.bert_ids(" ".join(self.src_words))
        self.tgt_bert_indice, self.tgt_segments_id, self.tgt_piece_id = self.tokenizer.bert_ids(" ".join(self.tgt_words))
        self.src_len = len(self.src_bert_indice)
        self.tgt_len = len(self.tgt_bert_indice)
    def __str__(self):
        ## print source words
        output = ' '.join(self.src_words) + '\n' + ' '.join(self.tgt_words) + '\n' + self.tag + '\n'
        return output


def parseInstance(texts, tokenizer):
    if len(texts) != 3: return None
    src_words, tgt_words = texts[0].strip().split(' '), texts[1].strip().split(' ')
    tag = texts[2].strip()

    return Instance(src_words, tgt_words, tag, tokenizer)

def writeInstance(filename, insts):
    with open(filename, 'w') as file:
        for inst in insts:
            file.write(str(inst) + '\n')

def printInstance(output, inst):
    output.write(str(inst) + '\n')
