import argparse
import os
import shutil
import regex

# noinspection PyUnresolvedReferences
from directory_iterator import all_files

# noinspection PyUnresolvedReferences
from bert_preprocess import preprocess


def to_bert_file(file, path, bert_dir):

    # Zatrzasnął	zatrzasnąć	0	praet:sg:m1:perf	O	O

    old_file = open(os.path.join(path, file), 'r', encoding='utf-8')

    new_file = open(os.path.join(bert_dir, file), 'w', encoding='utf-8')

    for line in old_file.readlines():
        if len(line) > 1:
            token, _, _, _, tag, _ = line[:-1].split('\t')
            new_line = '%s %s\n' % (token, tag)
        else:
            new_line = '\n'
        new_file.write(new_line)
    new_file.close()
    old_file.close()


if __name__ == '__main__':
    default_file = r'C:\Users\piotrek\Desktop\inf\magisterka\ner\data\training_datasets\wikipedia_filtered'
    default_tokenizer = r'bert-base-cased'
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', type=str, dest='file', default=default_file,
                        help='File or directory to replace contents')
    parser.add_argument('--tokenizer', '-t', type=str, dest='tokenizer', default=default_tokenizer,
                        help='Full path to tokenizer or name')
    args = parser.parse_args()
    print(args.tokenizer)

    max_len = 127
    path = args.file
    bert_path = os.path.join(path, 'bert')
    if not os.path.exists(bert_path):
        os.makedirs(bert_path)

    for file in os.listdir(path):
        if file.endswith('.tsv'):
            to_bert_file(file, path, bert_path)
            old_bert = os.path.join(bert_path, file)
            new_bert = os.path.join(bert_path, file[:-4] + '.txt')
            preprocess(old_bert, new_bert, args.tokenizer, max_len)
            os.remove(old_bert)
