
import argparse
import theano
import numpy
import cPickle as pkl

from nmt_word import (build_sampler, gen_sample, load_params, init_params, init_tparams)

from multiprocessing import Process, Queue




def translate_model(word_map0, queue, rqueue, pid, model, options, k, normalize, n_best):

    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    # allocate model parameters
    params = init_params(options)

    # load model parameters and set theano shared variables
    params = load_params(model, params)
    tparams = init_tparams(params)

    # word index
    f_init, f_next, f_lambda = build_sampler(tparams, options, trng, use_noise)

    def _translate(seq):
        xx = numpy.array(seq).reshape([len(seq), 1])
        word_map = list(set(list(xx.reshape(xx.shape[0]*xx.shape[1]))+word_map0))

        new_x_input = numpy.array([word_map.index(ii) for ii in xx.reshape(xx.shape[0]*xx.shape[1])]).reshape(xx.shape[0], xx.shape[1])
        sx_map = new_x_input.T.flatten()

        gen_x_mask = numpy.array([1 if jjj[0] !=0 else 0 for jjj in xx])
        # sample given an input sequence and obtain scores
        sample, score, _ = gen_sample(tparams, f_init, f_next, f_lambda,
                                   xx, gen_x_mask, sx_map, word_map,
                                   options, trng=trng, k=k, maxlen=200,
                                   stochastic=False, argmax=False)

        # normalize scores according to sequence lengths
        if normalize:
            lengths = numpy.array([len(s) for s in sample])
            score = score / lengths
        if n_best > 1:
            sidx = numpy.argsort(score)[:n_best]

        else:
            sidx = numpy.argmin(score)
        # return numpy.array(word_map)[sample[sidx]], numpy.array(score)[sidx]

        return numpy.array(sample)[sidx], numpy.array(score)[sidx]
        # return numpy.array(word_map)[sample[sidx]], numpy.array(score)[sidx]

    while True:
        req = queue.get()
        if req is None:
            break

        idx, x = req[0], req[1]
        print pid, '-', idx
        seq, scores = _translate(x)
        # print seq, scores

        rqueue.put((idx, seq, scores))

    # print tparams['att_lambda'].get_value()[0]

    return


def predict(model, dictionary, common_dictionary, source_file, saveto, k=5,
         normalize=False, n_process=5, chr_level=False, n_best=1):

    # load model model_options
    with open('%s.pkl' % model, 'rb') as f:
        options = pkl.load(f)

    # load source dictionary and invert
    with open(dictionary, 'rb') as f:
        word_dict = pkl.load(f)
    word_idict = dict()
    for kk, vv in word_dict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    word_idict_trg = word_idict
    # load target dictionary and invert
    # with open(dictionary_target, 'rb') as f:
    #     word_dict_trg = pkl.load(f)
    # word_idict_trg = dict()
    # for kk, vv in word_dict_trg.iteritems():
    #     word_idict_trg[vv] = kk
    # word_idict_trg[0] = '<eos>'
    # word_idict_trg[1] = 'UNK'

    word_map0 = []
    with open(common_dictionary) as ff:
        for line in ff:
            line = line.strip()
            if line in word_dict:
                if line not in word_map0:
                    word_map0.append(word_dict[line])



    # create input and output queues for processes
    queue = Queue()
    rqueue = Queue()
    processes = [None] * n_process
    for midx in xrange(n_process):
        processes[midx] = Process(
            target=translate_model,
            args=(word_map0, queue, rqueue, midx, model, options, k, normalize, n_best))
        processes[midx].start()

    # utility function
    def _seqs2words(caps):
        capsw = []
        for cc in caps:
            ww = []
            for w in cc:
                if w == 0:
                    break
                ww.append(word_idict_trg[w])
            capsw.append(' '.join(ww))
        return capsw
    # def _seqs2words(caps):
    #     capsw = []
    #     attw = []
    #     for cc in caps:
    #         ww = []
    #         www = []
    #         label = 0
    #         for w in cc:
    #             if w == 0 and label != 0:
    #                 break
    #             elif w == 0:
    #                 continue
    #             label += 1
    #             ww.append(word_idict_trg[w])
    #             www.append(str(tparams['att_lambda'].get_value()[w]))
    #         wwww = []
    #         for aa, bb in zip(ww, www):
    #             wwww.append(aa+'_'+bb)
    #         # capsw.append(' '.join(ww))
    #         capsw.append(' '.join(wwww))
    #     return capsw

    def _send_jobs(fname):
        with open(fname, 'r') as f:
            for idx, line in enumerate(f):
                if chr_level:
                    words = list(line.decode('utf-8').strip())
                else:
                    words = line.strip().split()
                x = map(lambda w: word_dict[w] if w in word_dict else 1, words)
                x = map(lambda ii: ii if ii < options['n_words'] else 1, x)
                x += [0]
                queue.put((idx, x))
        return idx+1

    def _finish_processes():
        for midx in xrange(n_process):
            queue.put(None)

    def _retrieve_jobs(n_samples):
        trans = [None] * n_samples
        scores = [None] * n_samples
        for idx in xrange(n_samples):
            resp = rqueue.get()
            trans[resp[0]] = resp[1]
            scores[resp[0]] = resp[2]
            if numpy.mod(idx, 10) == 0:
                print 'Sample ', (idx+1), '/', n_samples, ' Done'
        return trans, scores

    print 'Translating ', source_file, '...'
    n_samples = _send_jobs(source_file)
    trans, scores = _retrieve_jobs(n_samples)
    _finish_processes()

    if n_best == 1:
        trans = _seqs2words(trans)
    else:
        n_best_trans = []
        for idx, (n_best_tr, score_) in enumerate(zip(trans, scores)):
            sentences = _seqs2words(n_best_tr)
            for ids, trans_ in enumerate(sentences):
                n_best_trans.append(
                    '|||'.join(
                        ['{}'.format(idx), trans_,
                         '{}'.format(score_[ids])]))
        trans = n_best_trans

    with open(saveto, 'w') as f:
        print >>f, '\n'.join(trans)
    print 'Done'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=5, help="Beam size")
    parser.add_argument('-p', type=int, default=5, help="Number of processes")
    parser.add_argument('-n', action="store_true", default=False,
                        help="Normalize wrt sequence length")
    parser.add_argument('-c', action="store_true", default=False,
                        help="Character level")
    parser.add_argument('-b', type=int, default=1, help="Output n-best list")
    parser.add_argument('model', type=str)
    parser.add_argument('dictionary', type=str)
    parser.add_argument('common_dictionary', type=str)
    parser.add_argument('source', type=str)
    parser.add_argument('saveto', type=str)

    args = parser.parse_args()

    main(args.model, args.dictionary, args.common_dictionary, args.source,
         args.saveto, k=args.k, normalize=args.n, n_process=args.p,
         chr_level=args.c, n_best=args.b)
