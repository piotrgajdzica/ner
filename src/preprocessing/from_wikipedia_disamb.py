import os
import re
import sys
import traceback

import regex
import requests

sys.path.append("/".join(os.path.realpath(__file__).split('/')[:-2]))
from src.preprocessing.data_filter import DataFilter

from src.util.class_mapper import ClassMapper
from src.util.title_mapper import TitleMapper


def sentence_generator(paths):
    for path in paths:
        f = open(path, encoding='utf-8')
        for line in f.readlines():
            yield line


def batch_sentence_generator(paths, batch_size=100):
    sentences = []
    for sentence in sentence_generator(paths):
        sentences.append(sentence)
        if len(sentences) >= batch_size:
            yield sentences
            sentences = []
    yield sentences


def token_generator(sentence):
    tokens = sentence.split(' ')
    for token in tokens:
        # print(token)
        if token.startswith('___') and token.endswith('___'):
            # print(token.split('___'))
            _, token, page_id, _ = token.split('___')
        else:
            token, page_id = token, None

        begin = 'B'
        for token_part in token.split('_'):
            yield token_part, page_id, begin if page_id else None
            begin = 'I'

def clear_sentence(sentence):
    pat = re.compile('___(?P<token>.*?)___\d+___')

    def repl(m):
        return next(filter(None, m.groups()))

    return pat.sub(repl, sentence).replace('  ', ' ').replace('_', ' ')


def get_krnnt_tokens(sentences):
    content = requests.post('http://192.168.99.100:9003?output_format=conll', data=' '.join(sentences).encode('utf-8')).content.decode('utf-8')
    for line in content.split('\n'):
        if len(line) > 0:
            token, lemma, space, morph, *_ = line.split('\t')
            yield token, lemma, space, morph
        else:
            yield None, None, None, None


def get_krnnt_token_list(sentences):
    content = requests.post('http://192.168.99.100:9003?output_format=conll&input_format=lines', data='\n\n'.join(sentences).encode('utf-8')).content.decode('utf-8')
    for sentence in content.split('\n\n\n'):
        to_yield = []
        for line in sentence.split('\n'):
            if len(line) > 0:
                token, lemma, space, morph, *_ = line.split('\t')
                to_yield.append((token, lemma, space, morph))
            else:
                to_yield.append((None, None, None, None))
        yield to_yield


def tagged_sentence(sentence):
    for token, lemma, space, morph in get_krnnt_tokens([sentence]):
        yield token, lemma, space, morph


def tagged_sentences(sentences):
    for sentence in get_krnnt_token_list(sentences):
        yield sentence


def joined_tokens(tagged_tokens, annotated_tokens):
    ann_idx = 0
    ann_token, page_id, begin = annotated_tokens[ann_idx]
    token_batch = []
    for token, lemma, space, morph in tagged_tokens:
        if token is None:
            token_batch.append((None, None, None, None, None, None))
            yield token_batch
            token_batch = []
        else:
            while ann_token is None or token not in ann_token:
                ann_idx += 1
                ann_token, page_id, begin = annotated_tokens[ann_idx]
            ann_token = ann_token[ann_token.index(token):]
            token_batch.append((token, lemma, space, morph, page_id, begin))
    if token_batch:
        yield token_batch


def should_skip(token_batch) -> bool:

    if len(token_batch) > 60:
        return True

    pattern = regex.compile(r"^\p{Lu}")
    useful_tokens = 0

    for token, lemma, space, morph, page_id, begin in token_batch[1:]:
        if token is None:
            pass
        elif page_id is not None:
            wikidata_id = title_mapper[int(page_id)]['id']
            nkjp_class = entity_id_to_nkjp_class.get(wikidata_id, None)
            if nkjp_class is not None:
                useful_tokens += 1
        elif pattern.match(token):
            return True
    return useful_tokens == 0


if __name__ == '__main__':
    base_dir = r'C:\Users\piotrek\Desktop\inf\magisterka\ner'
    input_dir = os.path.join(base_dir, 'data/unfiltered_datasets/wikipedia-2017-dsmb')
    # file = 'wiki.dsmb.3886000-3953000.txt'
    annotated_tokens = []
    tagged_tokens = []
    full_page_id_cache_path = os.path.join(base_dir, 'data/cache/full_page_id_cache.json')

    corpus_output_dir = os.path.join(base_dir, 'data/training_datasets/wikipedia_disamb')
    test_path = os.path.join(base_dir, corpus_output_dir, 'test.tsv')
    dev_path = os.path.join(base_dir, corpus_output_dir, 'dev.tsv')
    train_path = os.path.join(base_dir, corpus_output_dir, 'train.tsv')
    test_file = open(test_path, 'a', encoding='utf-8')
    dev_file = open(dev_path, 'a', encoding='utf-8')
    train_file = open(train_path, 'a', encoding='utf-8')
    output_file_generator = DataFilter.target_set_generator({test_file: 1, dev_file: 1, train_file: 3})
    f = next(output_file_generator)

    title_mapper = TitleMapper.from_full_cache(full_page_id_cache_path)
    full_cache_path = os.path.join(base_dir, 'data/cache/full_cache.json')
    full_specific_cache_path = os.path.join(base_dir, 'data/cache/full_specific_cache.json')
    entity_id_to_nkjp_class = ClassMapper.from_full_cache(full_cache_path)
    entity_id_to_nkjp_specific_class = ClassMapper.from_full_cache(full_specific_cache_path)
    # self.page_id_to_wikidata_id[page_id] = {'id': wikidata_id, 'title': title}
    counter = 0
    f_counter_path = os.path.join(base_dir, 'data/unfiltered_datasets/last_counter.txt')
    f_counter = open(f_counter_path, 'r')
    last_counter = int(f_counter.read())
    print(last_counter)
    f_counter.close()
    for sentences in batch_sentence_generator([os.path.join(input_dir, file) for file in os.listdir(input_dir)]):
        counter += 1
        if counter <= last_counter:
            continue
        tagged_sentence_list = list(tagged_sentences([clear_sentence(sentence) for sentence in sentences]))
        for sentence_idx, sentence in enumerate(sentences):
            assert type(sentence_idx) == int
            try:
                annotated_tokens = []
                tagged_tokens = []
                for token, page_id, begin in token_generator(sentence):
                    annotated_tokens.append((token, page_id, begin))
                # for token, lemma, space, morph in tagged_sentence(clear_sentence(sentence)):
                for token, lemma, space, morph in tagged_sentence_list[sentence_idx]:
                    tagged_tokens.append((token, lemma, space, morph))
                # print(annotated_tokens)
                # print(tagged_tokens)
                # print(len(annotated_tokens))
                # print(len(tagged_tokens))
                for token_batch in joined_tokens(tagged_tokens, annotated_tokens):
                    if not should_skip(token_batch):
                        for token, lemma, space, morph, page_id, begin in token_batch:
                            if token is None:
                                f.write('\n')
                            else:
                                if page_id is not None:
                                    try:
                                        wikidata_id = title_mapper[int(page_id)]['id']
                                        nkjp_class = entity_id_to_nkjp_class.get(wikidata_id, None)
                                        nkjp_specific_class = entity_id_to_nkjp_specific_class.get(wikidata_id, None)
                                    except KeyError:
                                        nkjp_class = None
                                        nkjp_specific_class = None
                                else:
                                    nkjp_class = None
                                    nkjp_specific_class = None

                                if nkjp_class is None:
                                    begin = None
                                f.write('%s\t%s\t%s\t%s\t%s_%s\t%s_%s\n' % (
                                    token, lemma, space, morph, begin or '', nkjp_class or '',
                                    begin or '', nkjp_specific_class or ''))
                f = next(output_file_generator)
            except Exception:
                traceback.print_exc()

        print(counter)
        f_counter = open(f_counter_path, 'w')
        f_counter.write(str(counter))
        f_counter.close()

