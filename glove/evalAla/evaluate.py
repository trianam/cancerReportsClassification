import argparse
import numpy as np
import sys
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_file', default='vocab.txt', type=str)
    parser.add_argument('--vectors_file', default='vectors.txt', type=str)
    parser.add_argument('--output_file', default='evalTOP15.txt', type=str)
    parser.add_argument('--top_num', default=15, type=int)
    args = parser.parse_args()

    with open(args.vocab_file, 'r') as f:
        words = [x.rstrip().split(' ')[0] for x in f.readlines()]
    with open(args.vectors_file, 'r') as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            vectors[vals[0]] = list(map(float, vals[1:]))

    vocab_size = len(words)
    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}

    vector_dim = len(vectors[ivocab[0]])
    W = np.zeros((vocab_size, vector_dim))
    for word, v in vectors.items():
        if word == '<unk>':
            continue
        W[vocab[word], :] = v

    # normalize each word vector to unit variance
    W_norm = np.zeros(W.shape)
    d = (np.sum(W ** 2, 1) ** (0.5))
    W_norm = (W.T / d).T
    with open(args.output_file, 'w') as f:
        evaluate_vectors(W_norm, vocab, ivocab, args.top_num, f)

def evaluate_vectors(W, vocab, ivocab, top_num, outputFile):
    """Evaluate the trained word vectors on a variety of tasks"""

    filenames = [
        'benigno-tessuto.txt',
        'maligno-tessuto.txt',
        'benigno-maligno.txt',
        'morfologia-sede.txt',
        'gram8-plurali.txt',
        'gram8-plurali-short.txt'
        ]
    prefix = os.path.dirname(os.path.abspath(__file__))+'/question-data/'

    # to avoid memory overflow, could be increased/decreased
    # depending on system and vocab size
    split_size = 100

    correct_sem = 0; # count correct semantic questions
    correct_syn = 0; # count correct syntactic questions
    correct_tot = 0 # count correct questions
    count_sem = 0; # count all semantic questions
    count_syn = 0; # count all syntactic questions
    count_tot = 0 # count all questions
    full_count = 0 # count all questions, including those with unknown words

    for i in range(len(filenames)):
        with open('%s/%s' % (prefix, filenames[i]), 'r') as f:
            full_data = [line.rstrip().split(' ') for line in f]
            full_count += len(full_data)
            data = [x for x in full_data if all(word in vocab for word in x)]

        indices = np.array([[vocab[word] for word in row] for row in data])
        ind1, ind2, ind3, ind4 = indices.T

        predictions = np.zeros((len(indices),top_num))
        num_iter = int(np.ceil(len(indices) / float(split_size)))
        for j in range(num_iter):
            subset = np.arange(j*split_size, min((j + 1)*split_size, len(ind1)))

            pred_vec = (W[ind2[subset], :] - W[ind1[subset], :]
                +  W[ind3[subset], :])
            #cosine similarity if input W has been normalized
            dist = np.dot(W, pred_vec.T)
            #print(dist.shape)
            #sys.exit()

            for k in range(len(subset)):
                dist[ind1[subset[k]], k] = -np.Inf
                dist[ind2[subset[k]], k] = -np.Inf
                dist[ind3[subset[k]], k] = -np.Inf

            # predicted word index
            #predictions[subset] = np.argmax(dist, 0).flatten()
            predictions[subset] = np.argsort(dist, 0)[-top_num:].T
            #print(subset)
            #print(dist.shape)
            #print(predictions.shape)
            #sys.exit()
            #for ii in range(10):
            #    print("{} {}".format(ivocab[ind4[ii]], ivocab[predictions[ii]]))
            #sys.exit()

        #print(predictions.shape)
        #print(predictions[1:10])
        #print(ind4.shape)
        #print(predictions[0])
        #print(ind4[0])
        #sys.exit()
        #val = (ind4 == predictions) # correct predictions
        val = np.zeros(len(ind4))
        for j in range(len(val)):
            val[j] = np.any(predictions[j] == ind4[j])
        count_tot = count_tot + len(ind1)
        correct_tot = correct_tot + sum(val)
        if i < 5:
            count_sem = count_sem + len(ind1)
            correct_sem = correct_sem + sum(val)
        else:
            count_syn = count_syn + len(ind1)
            correct_syn = correct_syn + sum(val)

        print("%s:" % filenames[i])
        print('ACCURACY TOP%d: %.2f%% (%d/%d)' %
            (top_num, np.mean(val) * 100, np.sum(val), len(val)))

        outputFile.write("%s:\n" % filenames[i])
        outputFile.write('ACCURACY TOP%d: %.2f%% (%d/%d)\n' %
            (top_num, np.mean(val) * 100, np.sum(val), len(val)))

    print('Questions seen/total: %.2f%% (%d/%d)' %
        (100 * count_tot / float(full_count), count_tot, full_count))
    outputFile.write('Questions seen/total: %.2f%% (%d/%d)\n' %
        (100 * count_tot / float(full_count), count_tot, full_count))
    #print('Semantic accuracy: %.2f%%  (%i/%i)' %
    #    (100 * correct_sem / float(count_sem), correct_sem, count_sem))
    #print('Syntactic accuracy: %.2f%%  (%i/%i)' %
    #    (100 * correct_syn / float(count_syn), correct_syn, count_syn))
    print('Total accuracy: %.2f%%  (%i/%i)' % (100 * correct_tot / float(count_tot), correct_tot, count_tot))
    outputFile.write('Total accuracy: %.2f%%  (%i/%i)\n' % (100 * correct_tot / float(count_tot), correct_tot, count_tot))


if __name__ == "__main__":
    main()
