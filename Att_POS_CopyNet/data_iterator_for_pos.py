#-*- coding:utf-8 -*-
#######################################################
#
#    Author: Chuwei Luo
#    Email: luochuwei@gmail.com
#    Date: 03/08/2016
#    Usage: text iterator for pos
#
#######################################################

import cPickle as pkl
import gzip


def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)


class TextIterator:
    """Simple Bitext iterator."""
    def __init__(self, source, target, source_pos, target_pos,
                 word_dic, pos_dic,
                 batch_size=128,
                 maxlen=100,
                 n_words=-1,
                 n_pos=-1):
        self.source = fopen(source, 'r')
        self.source_pos = fopen(source_pos, 'r')
        self.target = fopen(target, 'r')
        self.target_pos = fopen(target_pos, 'r')
        with open(word_dic, 'rb') as f:
            self.word_dic = pkl.load(f)
        with open(pos_dic, 'rb') as f:
            self.pos_dic = pkl.load(f)

        self.batch_size = batch_size
        self.maxlen = maxlen

        self.n_words = n_words
        self.n_pos = n_pos

        self.end_of_data = False

    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)
        self.source_pos.seek(0)
        self.target.seek(0)
        self.target_pos.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        source_pos = []
        target = []
        target_pos = []

        try:

            # actual work here
            while True:

                # read from source file and map to word index
                ss = self.source.readline()
                if ss == "":
                    raise IOError
                ss = ss.strip().split()
                ss = [self.word_dic[w] if w in self.word_dic else 1
                      for w in ss]
                if self.n_words > 0:
                    ss = [w if w < self.n_words else 1 for w in ss]

                ssp = self.source_pos.readline()
                if ssp == "":
                    raise IOError
                ssp = ssp.strip().split()
                ssp = [self.pos_dic[w] if w in self.pos_dic else 1
                      for w in ssp]
                if self.n_pos > 0:
                    ssp = [w if w < self.n_pos else 1 for w in ssp]

                # read from source file and map to word index
                tt = self.target.readline()
                if tt == "":
                    raise IOError
                tt = tt.strip().split()
                tt = [self.word_dic[w] if w in self.word_dic else 1
                      for w in tt]
                if self.n_words > 0:
                    tt = [w if w < self.n_words else 1 for w in tt]

                ttp = self.target_pos.readline()
                if ttp == "":
                    raise IOError
                ttp = ttp.strip().split()
                ttp = [self.pos_dic[w] if w in self.pos_dic else 1
                      for w in ttp]
                if self.n_pos > 0:
                    ttp = [w if w < self.n_pos else 1 for w in ttp]

                if len(ss) > self.maxlen and len(tt) > self.maxlen and len(ssp) > self.maxlen and len(ttp) > self.maxlen:
                    continue

                source.append(ss)
                source_pos.append(ssp)
                target.append(tt)
                target_pos.append(ttp)

                if len(source) >= self.batch_size or \
                        len(target) >= self.batch_size or len(source_pos) >= self.batch_size or len(target_pos) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        if len(source) <= 0 or len(target) <= 0 or len(source_pos) <= 0 or len(target_pos) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return source, target, source_pos, target_pos
