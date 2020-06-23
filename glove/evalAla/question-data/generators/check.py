import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check_file', default='check.txt', type=str)
    parser.add_argument('--vectors_file', default='vectors.txt', type=str)
    args = parser.parse_args()

    with open(args.vectors_file, 'r') as f:
        words = [x.rstrip().split(' ')[0] for x in f.readlines()]

    sx = []
    dx = []
    with open(args.check_file, 'r') as f:
        for l in f.readlines():
            w = l.rstrip().split(' ')
            sx.append(w[0])
            dx.append(w[1])

    #check if we put doubles
    for w in sx:
        if sx.count(w) != 1 or dx.count(w) != 0:
            print("Multiple occurrences of: '{}'".format(w))
            return

        if words.count(w) == 0:
            print("Non existent word: '{}'".format(w))
            return

    for w in dx:
        if dx.count(w) != 1 or sx.count(w) != 0:
            print("Multiple occurrences of: '{}'".format(w))
            return

        if words.count(w) == 0:
            print("Non existent word: '{}'".format(w))
            return

    print("All ok")


if __name__ == "__main__":
    main()

