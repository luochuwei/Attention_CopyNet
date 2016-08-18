import numpy
import os
import cPickle

from Att_copy_s import train

def main(job_id, params):
    print params
    basedir = 'data_2'
    validerr = train(saveto=params['model'][0],
                                        reload_=params['reload'][0],
                                        dim_word=params['dim_word'][0],
                                        dim=params['dim'][0],
                                        n_words=params['n-words'][0],
                                        n_words_src=params['n-words'][0],
                                        decay_c=params['decay-c'][0],
                                        clip_c=params['clip-c'][0],
                                        lrate=params['learning-rate'][0],
                                        optimizer=params['optimizer'][0],
                                        maxlen=100,
                                        batch_size=32,
                                        valid_batch_size=32,
                    datasets=['%s/validation.s'%basedir,
                    '%s/validation.t'%basedir],
                    valid_datasets=['%s/validation.s'%basedir,
                    '%s/validation.t'%basedir,],
                    # dictionaries=['%s/p.txt.pkl'%basedir,
                    # '%s/r.txt.pkl'%basedir],
                    dictionaries=['%s/training.s.pkl'%basedir,'%s/commonwords.txt'%basedir],
                                        validFreq=1000,
                                        dispFreq=100,
                                        saveFreq=1000,
                                        sampleFreq=100,
                                        use_dropout=params['use-dropout'][0],
                                        overwrite=False,
                                        show_lambda=True)
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
    basedir = 'data_2'
    main(0, {
        'model': ['%s/model/m.npz'%basedir],
        'dim_word': [512],#word embedding dim
        'dim': [512],     #hidden dim
        'n-words': [10000],   #vocabulary size
        'optimizer': ['rmsprop'],
        'decay-c': [0.],
        'clip-c': [1.],
        'use-dropout': [False],
        'learning-rate': [0.05],
        'reload': [False]})


