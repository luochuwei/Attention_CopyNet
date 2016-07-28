#-*- coding:utf-8 -*-
##################################################################
#
#    Author: Chuwei Luo
#    Email: luochuwei@gmail.com
#    Date: 26/07/2016
#    Usage: Training
#
##################################################################

import numpy
import os
import cPickle

from Seq2SeqAtt import train

def main(job_id, params):
    print params
    basedir = 'data'
    validerr = train(saveto=params['model'][0], reload_=params['reload'][0], dim_word=params['dim_word'][0], dim=params['dim'][0], n_words=params['n-words'][0], n_words_src=params['n-words'][0], decay_c=params['decay-c'][0], clip_c=params['clip-c'][0], lrate=params['learning-rate'][0], optimizer=params['optimizer'][0], maxlen=15, batch_size=1, valid_batch_size=1, datasets=['%s/ppp.txt'%basedir, '%s/ppp.txt'%basedir], valid_datasets=['%s/pv.txt'%basedir,
                    '%s/pv.txt'%basedir],  dictionaries=['%s/p.txt.pkl'%basedir], validFreq=500000, dispFreq=1, saveFreq=100, sampleFreq=1, use_dropout=params['use-dropout'][0], overwrite=False)
    return validerr

if __name__ == '__main__':
    # f = cPickle.load(open(r'data//p.txt.pkl'))
    # print f
    """
    datasets:

    dictionaries:
    OrderedDict([('eos', 0), ('UNK', 1), ('b', 2), ('c', 3), ('a', 4)])
    OrderedDict([('eos', 0), ('UNK', 1), ('B', 2), ('C', 3), ('A', 4)])

    """
    basedir = 'data'
    main(0, {
        'model': ['%s/model/m.model'%basedir],
        'dim_word': [15],
        'dim': [24],
        'n-words': [6],
        'optimizer': ['rmsprop'],
        'decay-c': [0.],
        'clip-c': [1.],
        'use-dropout': [False],
        'learning-rate': [0.01],
        'reload': [False]})


