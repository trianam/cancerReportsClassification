import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default='input.txt', type=str)
    parser.add_argument('--output_file', default='output.txt', type=str)
    args = parser.parse_args()

    with open(args.input_file, 'r') as f:
        couples = [x.rstrip() for x in f.readlines()]

    with open(args.output_file, 'w') as f:
        for i in range(len(couples)):
            for j in [k for k in range(len(couples)) if k != i]:
                f.write("{} {}\n".format(couples[i], couples[j]))
            

if __name__ == "__main__":
    main()

