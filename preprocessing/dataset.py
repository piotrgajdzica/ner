import os
import time
from random import random

from preprocessing.class_mapper import ClassMapper


def line_batch_generator(path, batch_size=100000, limit_size=None):
    f = open(path, encoding='utf-8')
    counter = 0
    while True:
        lines = f.readlines(batch_size)
        print(counter)
        if len(lines) == 0:
            return
        for line in lines:
            counter += 1
            if limit_size is not None and counter > limit_size:
                return
            yield line


def target_set_generator(d):
    s = sum(d.values())
    while True:
        for key in d:
            if random() < d[key] / s:
                yield key



def prepare_dataset(limit_size=None):
    full_cache_path = os.path.join('..', 'data', 'full_cache.json')
    entities_path = os.path.join('..', 'data', 'entities.jsonl')
    corpus_path = os.path.join('..', 'data', 'tokens-with-entities-and-tags.tsv')
    test_path = os.path.join('..', 'data', 'tokens-with-entities-tags-and-classes_sample2', 'test.tsv')
    dev_path = os.path.join('..', 'data', 'tokens-with-entities-tags-and-classes_sample2', 'dev.tsv')
    train_path = os.path.join('..', 'data', 'tokens-with-entities-tags-and-classes_sample2', 'train.tsv')

    if not os.path.isfile(full_cache_path):
        mapper = ClassMapper.from_entities(entities_path)
        mapper.build_nkjp_cache_and_store(full_cache_path)
    entity_id_to_nkjp_class = ClassMapper.from_full_cache(full_cache_path)
    test_file = open(test_path, 'w', encoding='utf-8')
    dev_file = open(dev_path, 'w', encoding='utf-8')
    train_file = open(train_path, 'w', encoding='utf-8')
    start = time.time()
    generator = line_batch_generator(corpus_path, limit_size=limit_size)
    output_file_generator = target_set_generator({test_file: 1, dev_file: 1, train_file: 3})
    target_file = next(output_file_generator)

    last_entity = None
    last_article_no = None
    is_article_title = False
    sentence_counter = 0
    article_skip = False
    for line in generator:
        columns = line[:-1].split('\t')
        if len(columns) == 7:
            nkjp_class = None
            article_no, word, base, space, tags, entity, entity_wikidata_id = columns
            if last_article_no != article_no:
                article_skip = False
                target_file = next(output_file_generator)
                is_article_title = True
            elif article_skip:
                continue
            last_article_no = article_no
            if word == '–':
                is_article_title = False
                continue
            if is_article_title:
                continue
                # we skip the titles
                # entity_wikidata_id = 'Q%s' % article_no
            if entity_wikidata_id != '_':
                entity_wikidata_id = int(entity_wikidata_id[1:])
                nkjp_class = entity_id_to_nkjp_class.get(entity_wikidata_id)
                # if nkjp_class is not None:
                #     print(word, entity, nkjp_class)
            start_tag = ''
            if nkjp_class:
                start_tag = 'B' if last_entity != entity else 'I'
            target_file.write('%s\t%s\t%s\t%s\t%s_%s\n' % (word, base, space, tags, start_tag, nkjp_class or ''))
            last_entity = entity
        else:
            if not article_skip:
                target_file.write(line)
                sentence_counter += 1
            if sentence_counter >= 3:
                sentence_counter = 0
                article_skip = True
    print(time.time() - start)


if __name__ == '__main__':
    prepare_dataset(limit_size=10000000)

