import csv
import re
import sys


def main():
    lst = []
    abbrs = {}
    with open(sys.argv[1], "r", encoding="cp1252") as f:
        for line in f:
            line_split = line.split('|')
            assert len(line_split) == 7
            abbr, label = line_split[0], line_split[1]
            if label == 'NAME' or label == 'UNSURED SENSE' or label.startswith('MISTAKE:'):
                continue
            if label == 'GENERAL ENGLISH':
                label = abbr
                line_split[1] = abbr
            mat = re.match(r'^(.+):(.+?)$', label)
            if mat:
                abbr = mat[2]
                label = mat[1]
                line_split[2] = mat[2]
                line_split[1] = mat[1]
            lst.append(line_split)
            if abbr not in abbrs:
                abbrs[abbr] = set()
            abbrs[abbr].add(label)
    cnt = 0
    num_negs = max(len(v) for k, v in abbrs.items()) - 1
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter='\t', quotechar=None)
        writer.writerow(['index', 'group', 'left', 'right', 'label'] + [f'neg_{i}' for i in range(num_negs)])
        for line_split in lst:
            text = line_split[6].rstrip()
            pos_l, pos_r = int(line_split[3]), int(line_split[4])
            left, right = text[:pos_l], text[pos_r + 1:]
            left = re.sub(r'_%#.*?#%_', ' ', left)
            right = re.sub(r'_%#.*?#%_', ' ', right)
            negs = [w for w in abbrs[line_split[0]] if w != line_split[1]]
            if not negs:
                continue
            while len(negs) <= num_negs:
                negs.append(' ')
            writer.writerow([cnt, line_split[0], left, right, line_split[1]] + negs)
            cnt += 1


if __name__ == '__main__':
    main()
