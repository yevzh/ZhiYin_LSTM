import argparse
import jieba
import pandas as pd


def load_stops(file_name):
    dev_list = []
    with open(file_name, encoding='utf8') as f:
        dev = f.readline()
        while dev:
            dev_list.append(dev[:-1])
            dev = f.readline()
    return dev_list


def ccft_comment(f):
    cache = f.readline()
    while cache and ((cache[0] == '\n') or ('<' in cache)):
        cache = f.readline()
    if not cache:
        return None
    comment = cache.replace('\n', '')
    cache = f.readline()
    while cache and (cache[0] != '<'):
        comment += cache.replace('\n', '')
        cache = f.readline()
    return comment


def ccft_gets(dev_list, file_names, seg):
    inf_list = [['labels', 'comments']]
    for file_name in file_names:
        label = '0'
        if 'positive' in file_name:
            label = '1'
        with open(file_name, encoding='utf8') as f:
            inf = ccft_comment(f)
            while inf:
                for dev in dev_list:
                    while dev in inf:
                        inf = inf.replace(dev, ' ')

                if seg:
                    inf = ' '.join(list(jieba.cut(inf, cut_all=False)))
                inf = ' '.join(inf.split())
                if inf:
                    inf_list.append([label, inf])
                inf = ccft_comment(f)
    return inf_list


def get_comments(dev_list, file_names, labeled, seg):
    inf_list = [['labels', 'comments']]
    for file_name in file_names:
        with open(file_name, encoding='utf8') as f:
            inf = f.readline()[1:-1]
            while inf:
                if labeled:
                    label = inf[0]
                    inf = inf[1:]
                for dev in dev_list:
                    while dev in inf:
                        inf = inf.replace(dev, ' ')

                if seg:
                    inf = ' '.join(list(jieba.cut(inf, cut_all=False)))
                inf = ' '.join(inf.split())
                if inf:
                    if labeled:
                        inf_list.append([label, inf])
                    else:
                        inf_list.append(['0', inf])
                inf = f.readline()
    return inf_list


def write_csv(inf_list, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        for inf in inf_list:
            f.write(f'{inf[0]},{inf[1]}\n')


def write_xml(inf_list, file_name):
    if inf_list:
        inf_list = inf_list[1:]
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write('<weibos>\n')
        count = 0
        for inf in inf_list:
            count += 1
            f.write(f'\t<weibo id="{count}" polarity="{inf[0]}">' + inf[1] + '</weibo>\n')
        f.write('</weibos>\n')


def config_setting():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--trans_type', type=str, default='cn', choices=['cn', 'en'])
    parser.add_argument('-i', '--input_file', type=str, default='ccft')
    parser.add_argument('-o', '--output_file', type=str, default='ccft')
    parser.add_argument('-s', '--segment', type=bool, default=True)
    args = parser.parse_args()
    if args.trans_type == 'en':
        args.segment = False
    path = 'Input'
    if args.input_file == 'ccft':
        infiles = [path + f'\\{args.trans_type}sample.positive.txt', path + f'\\{args.trans_type}sample.negative.txt']
    else:
        infiles = [(path + '\\' + filename) for filename in args.input_file.split()]
    path = 'DataSet'
    if args.output_file == 'ccft':
        outfile = path + f'\\ccft_{args.trans_type}.csv'
    else:
        outfile = path + '\\' + args.output_file
    return args, infiles, outfile


if __name__ == '__main__':
    args, infiles, outfile = config_setting()
    stop_list = load_stops(f'Corpus\\{args.trans_type}_stopwords.txt')
    if args.input_file == 'ccft':
        comment_list = ccft_gets(stop_list, infiles, args.segment)
    else:
        comment_list = get_comments(stop_list, infiles, 0, args.segment)
    write_csv(comment_list, outfile)
