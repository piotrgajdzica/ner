import os
import random


def sentences(f):
    sentence = []
    while True:
        lines = f.readlines(1000000)
        if len(lines) == 0:
            if sentence:
                yield sentence
            return
        for line in lines:
            sentence.append(line)
            if len(line) <= 2:
                yield sentence
                sentence = []


def downsample(directory: str, downsample: float):
    downsample_dir = os.path.join(directory, str(downsample))
    try:
        os.makedirs(downsample_dir)
    except FileExistsError:
        pass

    for file in os.listdir(directory):
        if file.endswith('.tsv') or file.endswith('.txt') and not os.path.isfile(os.path.join(downsample_dir, file)):
            f_in = open(os.path.join(directory, file), 'r', encoding='utf-8')
            f_out = open(os.path.join(downsample_dir, file), 'w', encoding='utf-8')
            for sentence in sentences(f_in):
                if random.random() < downsample:
                    for line in sentence:
                        f_out.write(line)
            f_in.close()
            f_out.close()
    return downsample_dir


if __name__ == '__main__':
    base_dir = r"""C:\Users\piotrek\Desktop\inf\magisterka\ner"""
    downsample(os.path.join(base_dir, 'data/training_datasets/wikipedia_disamb'), 0.1)
