import os
from collections import OrderedDict

import regex
import nltk

from preprocessing.directory_iterator import all_files


def token_matches(text, tag, start_idx):
    text_len = len(text)
    for idx in range(len(tag)):
        # print(text[start_idx + idx], tag[idx])
        if start_idx + idx >= text_len or text[start_idx + idx][0] != tag[idx]:
            return False
    return True

def preprocess_line(line, annotations):
    annotated = [(token, 'O') for token in line]
    # print(annotated)

    for text_annotation, tag_annotation in annotations.items():
        # print(text_annotation, tag_annotation)
        for idx in range(len(annotated)):
            token = annotated[idx][0]
            tag = annotated[idx][1]
            if token_matches(annotated, text_annotation, idx):
                annotated[idx] = annotated[idx][0], 'B-' + tag_annotation
                for tag_idx in range(1, len(text_annotation)):
                    annotated[idx + tag_idx] = annotated[idx + tag_idx][0], 'I-' + tag_annotation
    return annotated


def preprocess_file(annotated):
    raw = annotated.replace('annotated', 'raw').replace('out', 'txt')

    annotated_text = open(annotated, 'r', encoding='utf-8').readlines()[1:]
    raw_lines = open(raw, 'r', encoding='utf-8').readlines()
    annotations = {}
    for line in annotated_text:
        tag, lemma, category, specific_category = line.split('\t')
        annotations[tuple(nltk.word_tokenize(tag, language='polish'))] = category
    annotations = OrderedDict(sorted(annotations.items(), key=lambda t: len(t[0])))
    nltk_tokens = [nltk.word_tokenize(line, language='polish') for line in raw_lines[4:]]
    for token_line in nltk_tokens:
        if len(token_line) > 0:
            line = preprocess_line(token_line, annotations)
            for el in line:
                yield "%s %s" % el
            yield ""


if __name__ == '__main__':

    # word_data = "Ala ma kota, ale nie mieć psa."
    # nltk_tokens = nltk.word_tokenize(word_data, language='polish')
    # print(nltk_tokens)
    # exit(0)
    # s = "Ala ma kota, ale nie mieć psa."
    # print(regex.findall("[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+", s))
    # exit(0)

    dataset_mapping = {
        'asia': 'train',
        'nord_stream': 'dev',
        'ryanair': 'test'
    }

    base_directory = r'C:\Users\piotrek\Desktop\inf\magisterka\ner\data\unfiltered_datasets\bsnlp'
    base_output_directory = r'C:\Users\piotrek\Desktop\inf\magisterka\ner\data\training_datasets\bsnlp\bert'

    for source, destination in dataset_mapping.items():
        output_file = open(os.path.join(base_output_directory, destination + '.txt'), 'w', encoding='utf-8')
        for file in all_files(os.path.join(base_directory, source)):
            # print(file, output_file)
            if 'annotated' in file:
                for line in preprocess_file(file):
                    output_file.write("%s\n" % line)

