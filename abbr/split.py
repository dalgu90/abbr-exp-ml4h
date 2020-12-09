import csv
import math
import os
import sys
from collections import OrderedDict

import numpy as np


def main():
    groups = OrderedDict()
    np.random.seed(1)
    valid_ratio = float(sys.argv[3])
    test_ratio = float(sys.argv[4])
    train_ratio = 1.0 - valid_ratio - test_ratio
    assert 0 <= train_ratio <= 1.0
    assert 0 <= valid_ratio <= 1.0
    assert 0 <= test_ratio <= 1.0
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
    with open(os.path.join(sys.argv[2], 'train.tsv'), "w", encoding="utf-8") as f_train, open(
            os.path.join(sys.argv[2], 'dev.tsv'), 'w', encoding='utf-8') as f_dev, open(
        os.path.join(sys.argv[2], 'test.tsv'), 'w', encoding='utf-8') as f_test:
        writer_train = csv.writer(f_train, delimiter='\t', quotechar=None)
        writer_train.writerow(['index', 'group', 'left', 'right', 'label'] + [f'neg_{i}' for i in range(num_negs)])
        writer_dev = csv.writer(f_dev, delimiter='\t', quotechar=None)
        writer_dev.writerow(['index', 'group', 'left', 'right', 'label'] + [f'neg_{i}' for i in range(num_negs)])
        writer_test = csv.writer(f_test, delimiter='\t', quotechar=None)
        writer_test.writerow(['index', 'group', 'left', 'right', 'label'] + [f'neg_{i}' for i in range(num_negs)])
        for group, lst in groups.items():
            np.random.shuffle(lst)
            lst_train = sorted(lst[:math.ceil(len(lst) * train_ratio)])
            lst_dev = sorted(lst[math.ceil(len(lst) * train_ratio):math.ceil(len(lst) * (1.0 - test_ratio))])
            lst_test = sorted(lst[math.ceil(len(lst) * (1.0 - test_ratio)):])
            for line in lst_train:
                writer_train.writerow(line)
            for line in lst_dev:
                writer_dev.writerow(line)
            for line in lst_test:
                writer_test.writerow(line)


if __name__ == '__main__':
    main()
