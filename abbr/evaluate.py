import argparse
import os
import pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="output path", type=str)
    parser.add_argument("--lenpen", help="length penalty", type=float, default=1.0)
    parser.add_argument("--loss_similarity", help="weight on similarity loss", type=float, default=0.1)
    args = parser.parse_args()

    with open(os.path.join(args.path, 'output.pkl'), 'rb') as f:
        m = pickle.load(f)
    grouped_res = {}
    c, t = 0, 0
    for i, e in enumerate(m):
        if args.loss_similarity > 0.0:
            pos = (e['log_probs_label'].sum() / (len(e['log_probs_label']) ** args.lenpen) +
                args.loss_similarity * e['inner_label'])
            negs = [v.sum() / (len(v) ** args.lenpen) + args.loss_similarity * e['inner_neg_' + k.split('_')[-1]]
                    for k, v in e.items() if k.startswith('log_probs_neg')]
        else:
            pos = e['log_probs_label'].sum() / (len(e['log_probs_label']) ** args.lenpen)
            negs = [v.sum() / (len(v) ** args.lenpen) for k, v in e.items() if k.startswith('log_probs_neg')]
        c0 = int((not negs) or (pos >= max(negs)))

        c += c0
        t += 1

        if e['group'] not in grouped_res:
            grouped_res[e['group']] = [c0, 1]
        else:
            grouped_res[e['group']][0] += c0
            grouped_res[e['group']][1] += 1

    print(f'acc (micro): {c / t}')
    print(f'acc (macro): {sum(v1 / v2 for k, (v1, v2) in grouped_res.items()) / len(grouped_res)}')


if __name__ == '__main__':
    main()
