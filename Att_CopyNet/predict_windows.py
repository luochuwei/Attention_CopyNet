#-*- coding:utf-8 -*-
############################################
#
#    Author: Chuwei Luo
#    Email: luochuwei@gmail.com
#    Date: 15/08/2016
#    Usage: for testing in Windows
#
############################################
import translate as TTT

if __name__ == '__main__':
    TTT.predict(r'data_2/model/m.npz', r'data_2/word_dict.pkl', r'data_2/dict2.txt', r'data_2/p.txt', r'data_2/ttt.txt', k=5, n_process=1)


