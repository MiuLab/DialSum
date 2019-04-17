import numpy as np

def loadVocabulary(path):
    if not isinstance(path, str):
        raise TypeError('path should be a string')

    vocab = []
    rev = []

    with open(path) as fd:
        for line in fd:
            line = line.rstrip('\r\n')
            rev.append(line)
        vocab = dict([(x,y) for (y,x) in enumerate(rev)])

    return {'vocab': vocab, 'rev': rev}

def sentenceToIds(data, vocab):
    if not isinstance(vocab, dict):
        raise TypeError('vocab should be a dict that contains vocab and rev')
    vocab = vocab['vocab']
    if isinstance(data, str):
        if data.find('<EOS>') != -1: #input sequence case
            tmp = data.split('<EOS>')[:-1]
            words = []
            for i in tmp:
                words.append(i.split())
        else:
            words = data.split()
    elif isinstance(data, list):
        raise TypeError('list type data is not implement yet')
        words = data
    else:
        raise TypeError('data should be a string or a list contains words')

    ids = []
    for w in words:
        if isinstance(w,list): #input sequence case
            sent = []
            for i in w:
                if str.isdigit(i) == True:
                    i = '0'
                sent.append(vocab.get(i, vocab['_UNK']))
            ids.append(sent)
        else:
            if str.isdigit(w) == True:
                w = '0'
            ids.append(vocab.get(w, vocab['_UNK']))
    return ids

def padSentence(s, max_length, vocab, word_in_sent_length=0):
    if isinstance(s[0],list): #input sequence case
        for _ in range(max_length-len(s)):
            s.append([vocab['vocab']['_PAD']]*word_in_sent_length)
        return s
    else:
        return s + [vocab['vocab']['_PAD']]*(max_length - len(s))

def computeAccuracy(correct_das, pred_das):
    correctChunkCnt = 0
    foundPredCnt = 0
    for correct_da, pred_da in zip(correct_das, pred_das):
        for c, p in zip(correct_da, pred_da):
            correctTag = c
            predTag = p
            if predTag == correctTag:
                correctChunkCnt += 1
            foundPredCnt += 1

    if foundPredCnt > 0:
        precision = 100*correctChunkCnt/foundPredCnt
    else:
        precision = 0

    return precision

class DataProcessor(object):
    def __init__(self, in_path, da_path, sum_path, in_vocab, da_vocab):
        self.__fd_in = open(in_path, 'r')
        self.__fd_da = open(da_path, 'r')
        self.__fd_sum = open(sum_path, 'r')
        self.__in_vocab = in_vocab
        self.__da_vocab = da_vocab
        self.end = 0

    def close(self):
        self.__fd_in.close()
        self.__fd_da.close()
        self.__fd_sum.close()

    def get_batch(self, batch_size):
        in_data = []
        da_data = []
        da_weight = []
        length = []
        sum_data = []
        sum_weight = []
        sum_length = []

        batch_in = []
        batch_da = []
        batch_sum = []
        max_len = 0
        max_sum_len = 0
        max_word_in_sent = 0

        #used to record word(not id)
        in_seq = []
        da_seq = []
        sum_seq = []
        for i in range(batch_size):
            inp = self.__fd_in.readline()
            if inp == '':
                self.end = 1
                break
            da = self.__fd_da.readline()
            summ = self.__fd_sum.readline()
            inp = inp.rstrip()
            da = da.rstrip()
            summ = summ.rstrip()

            in_seq.append(inp)
            da_seq.append(da)
            sum_seq.append(summ)

            inp = sentenceToIds(inp, self.__in_vocab)
            da = sentenceToIds(da, self.__da_vocab)
            summ = sentenceToIds(summ, self.__in_vocab)
            batch_in.append(np.array(inp))
            batch_da.append(np.array(da))
            batch_sum.append(np.array(summ))
            length.append(len(inp))
            sum_length.append(len(summ))
            if len(inp) > max_len:
                max_len = len(inp)
            if len(summ) > max_sum_len:
                max_sum_len = len(summ)
            if len(max(inp,key=len)) > max_word_in_sent:
                max_word_in_sent = len(max(inp,key=len))

        length = np.array(length)
        sum_length = np.array(sum_length)

        for i, s, ints in zip(batch_in, batch_da, batch_sum):
            a = []
            for sent in i:
                a.append(padSentence(list(sent), max_word_in_sent, self.__in_vocab))
            in_data.append(padSentence(list(a), max_len, self.__in_vocab, max_word_in_sent))
            da_data.append(padSentence(list(s), max_len, self.__da_vocab))
            sum_data.append(padSentence(list(ints), max_sum_len, self.__in_vocab))
        in_data = np.array(in_data)
        da_data = np.array(da_data)
        sum_data = np.array(sum_data)

        for s in da_data:
            weight = np.not_equal(s, np.zeros(s.shape))
            weight = weight.astype(np.float32)
            da_weight.append(weight)
        da_weight = np.array(da_weight)
        for i in sum_data:
            weight = np.not_equal(i, np.zeros(i.shape))
            weight = weight.astype(np.float32)
            sum_weight.append(weight)
        sum_weight = np.array(sum_weight)

        return in_data, da_data, da_weight, length, sum_data, sum_weight, sum_length, in_seq, da_seq, sum_seq

