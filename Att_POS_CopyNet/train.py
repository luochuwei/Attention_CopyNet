import numpy
import os
import cPickle

from nmt_new_pos_word import train

def main(job_id, params):
    print params
    basedir = 'data_2'
    validerr = train(saveto=params['model'][0],
                                        reload_=params['reload'][0],
                                        dim_word=params['dim_word'][0],
                                        dim_pos=params['dim_pos'][0],
                                        dim=params['dim'][0],
                                        n_words=params['n-words'][0],
                                        n_pos=params['n-pos'][0]+1,
                                        n_words_src=params['n-words'][0],
                                        decay_c=params['decay-c'][0],
                                        clip_c=params['clip-c'][0],
                                        lrate=params['learning-rate'][0],
                                        optimizer=params['optimizer'][0],
                                        maxlen=15,
                                        batch_size=4,
                                        valid_batch_size=1,
                    datasets=['%s/p.txt'%basedir,
                    '%s/p.txt'%basedir,
                    '%s/p_pos.txt'%basedir,
                    '%s/p_pos.txt'%basedir],
                    valid_datasets=['%s/p.txt'%basedir,
                    '%s/p.txt'%basedir,
                    '%s/p_pos.txt'%basedir,
                    '%s/p_pos.txt'%basedir],
                    # dictionaries=['%s/p.txt.pkl'%basedir,
                    # '%s/r.txt.pkl'%basedir],
                    dictionaries=['%s/word_dict.pkl'%basedir,'%s/pos_dict.pkl'%basedir,'%s/dict2.txt'%basedir],
                                        validFreq=50000,
                                        dispFreq=1,
                                        saveFreq=100,
                                        sampleFreq=1,
                                        use_dropout=params['use-dropout'][0],
                                        overwrite=False)
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
        'dim_word': [100],#word embedding dim
        'dim_pos': [100], #pos embedding dim
        'dim': [100],     #hidden dim
        'n-words': [6],   #vocabulary size
        'n-pos':[6],      #pos tag set size
        'optimizer': ['rmsprop'],
        'decay-c': [0.],
        'clip-c': [1.],
        'use-dropout': [False],
        'learning-rate': [0.01],
        'reload': [False]})


