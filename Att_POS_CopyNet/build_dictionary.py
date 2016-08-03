import numpy
import cPickle as pkl

import sys
import fileinput

from collections import OrderedDict

def main(f_list, dictname, is_pos_dict=False):
    word_freqs = OrderedDict()
    for filename in f_list:
        print 'Processing', filename
        with open(filename, 'r') as f:
            for line in f:
                words_in = line.strip().split(' ')
                for w in words_in:
                    if w not in word_freqs:
                        word_freqs[w] = 0
                    else:
                        word_freqs[w] += 1
    words = word_freqs.keys()
    freqs = word_freqs.values()

    sorted_idx = numpy.argsort(freqs)
    sorted_words = [words[ii] for ii in sorted_idx[::-1]]

    worddict = OrderedDict()
    worddict['eos'] = 0
    worddict['UNK'] = 1
    kk = 2
    if is_pos_dict:
        worddict = OrderedDict()
        worddict['eos'] = 0
        kk=1

    for ii, ww in enumerate(sorted_words):
        worddict[ww] = ii+kk

    pkl.dump(worddict, open('data_2/%s.pkl'%dictname, 'wb'), True)
    print worddict

    print 'Done'

if __name__ == '__main__':
    f_list1 = ['data_2/p_pos.txt', 'data_2/r_pos.txt']
    main(f_list1, 'pos_dict', is_pos_dict=True)

    f_list2 = ['data_2/p.txt', 'data_2/r.txt']
    main(f_list2, 'word_dict')

