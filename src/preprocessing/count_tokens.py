import os
import re


if __name__ == '__main__':
    path = r'C:\Users\piotrek\Desktop\inf\magisterka\ner\data\training_datasets\wikipedia_disamb'
    files = ['dev.tsv', 'train.tsv', 'test.tsv']
    # files = ['test.tsv']
    pattern = r'\n(\s*\n)*'
    s = 0
    for f in [open(os.path.join(path, file), encoding='utf-8') for file in files]:
        s += len(re.findall(pattern, f.read()))
    print(s)
