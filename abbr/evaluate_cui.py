import argparse
import os
import pickle
import csv
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="output path", type=str)
    parser.add_argument("--lenpen", help="length penalty", type=float, default=1.0)
    parser.add_argument("--test_file_path", help="path to the test file", type=str)
    args = parser.parse_args()

    lines = []
    with open(args.test_file_path, 'r') as f:
        reader = csv.reader(f, delimiter="\t", quotechar=None)
        for line in reader:
            lines.append(line)
    examples = {}
    for i, line in enumerate(lines):
        if i == 0:
            continue
        guid = int(line[0])
        label_cui = line[4]
        train_cuis = [t for t in line[5:] if t.strip()]
        examples[guid] = {'guid': guid, 'label_cui': label_cui, 'train_cuis': train_cuis}

    with open(os.path.join(args.path, 'output.pkl'), 'rb') as f:
        m = pickle.load(f)
    grouped_res = {}
    c, t = 0, 0
    for i, e in enumerate(m):
        guid = e['guid']
        label_cui = examples[guid]['label_cui']
        train_cuis = examples[guid]['train_cuis']
        train_log_probs = [e[f'log_probs_train_{i}'].sum() / len(e[f'log_probs_train_{i}']) ** args.lenpen \
                           for i in range(len(train_cuis))]
        output_cui = train_cuis[np.argmax(train_log_probs)]

        c0 = int(output_cui == label_cui)

        c += c0
        t += 1

        if e['group'] not in grouped_res:
            grouped_res[e['group']] = [c0, 1]
        else:
            grouped_res[e['group']][0] += c0
            grouped_res[e['group']][1] += 1

    print(f'acc (micro): {c / t} = {c}/{t}')
    print(f'acc (micro all): {c / len(examples)} = {c}/{len(examples)}')
    print(f'acc (macro): {sum(v1 / v2 for k, (v1, v2) in grouped_res.items()) / len(grouped_res)}')

if __name__ == '__main__':
    main()
