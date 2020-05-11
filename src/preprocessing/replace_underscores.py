import argparse
import os
import shutil
import regex

from .directory_iterator import all_files


def replace_underscare_in_file(file):
    backup = file + '.bakcup'
    shutil.move(file, backup)

    new_file = open(file, 'w', encoding='utf-8')
    text = regex.sub(r"(?<=\s+)_(?=\s+)", r"O", open(backup, encoding='utf-8').read())
    print(text[:1000])
    new_file.write(text)
    os.remove(backup)


if __name__ == '__main__':
    default_file = r'C:\Users\piotrek\Desktop\inf\magisterka\ner\data\training_datasets\wikipedia_disamb\bert'
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', type=str, dest='file', default=default_file,
                        help='File or directory to replace contents')
    args = parser.parse_args()

    path = args.file

    for file in all_files(path):
        if file.endswith('.tsv') or file.endswith('.txt'):
            replace_underscare_in_file(file)
