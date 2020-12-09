import csv
import math
import os
import sys
from collections import OrderedDict

import numpy as np


def main():
    # Read input
    groups = OrderedDict()
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter='\t', quotechar=None)
        for i, line in enumerate(reader):
            if i == 0:
                assert line[:5] == ['index', 'group', 'left', 'right', 'label'], line[:5]
                assert all(s.startswith('neg_') for s in line[5:])
                num_negs = len(line) - 5
                continue
            line[0] = int(line[0])
            if line[1] not in groups:
                groups[line[1]] = [line]
            else:
                groups[line[1]].append(line)

    # Split into 10 groups
    np.random.seed(1)
    for group, lst in groups.items():
        np.random.shuffle(lst)
        new_splits = [[] for _ in range(10)]
        for i, line in enumerate(lst):
            new_splits[i%10].append(line)
        groups[group] = new_splits

    # CV outputs
    cv_dir = os.path.join(sys.argv[2], 'cv')
    if not os.path.exists(cv_dir):
        os.makedirs(cv_dir)
    for cv_idx in range(10):
        with open(os.path.join(cv_dir, f'{cv_idx}.tsv'), 'w', encoding='utf-8') as fd:
            writer = csv.writer(fd, delimiter='\t', quotechar=None)
            writer.writerow(['index', 'group', 'left', 'right', 'label'] + [f'neg_{i}' for i in range(num_negs)])
            for group, splits in groups.items():
                for line in splits[cv_idx]:
                    writer.writerow(line)

if __name__ == '__main__':
    main()
