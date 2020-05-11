import os
import time
from random import random
from typing import Set

from src.preprocessing.dataset import Article
from src.util.class_mapper import ClassMapper
from src.util.title_mapper import TitleMapper


class DataFilter:
    def __init__(self, input_directory, output_directory, base_project_dir):
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.base_project_dir = base_project_dir
        self.articles: Set[Article] = set()
        self.entity_id_to_nkjp_class = None
        self.entity_id_to_nkjp_specific_class = None
        self.page_id_to_wikidata_id = None

    def set_up(self):
        raise NotImplementedError('You need to implement set_up method')

    def process_line(self, line: str):
        raise NotImplementedError('You need to implement process_line method')

    def filter_articles(self):
        raise NotImplementedError('You need to implement filter_articles method')

    @staticmethod
    def line_batch_generator(path, batch_size=1000000):
        def process_file(file):
            f = open(os.path.join(path, file), encoding='utf-8')
            counter = 0
            while True:
                lines = f.readlines(batch_size)
                print(counter)
                if len(lines) == 0:
                    return
                for line in lines:
                    counter += 1
                    yield line

        if os.path.isdir(path):
            for file in os.listdir(path):
                for el in process_file(file):
                    yield el
        if os.path.isfile(path):
                for el in process_file(path):
                    yield el

    @staticmethod
    def target_set_generator(d):
        s = sum(d.values())
        while True:
            for key in d:
                if random() < d[key] / s:
                    yield key

    def filter_data_and_save(self, split_weights=(1, 1, 3), line_limit=None):
        line_counter = 0
        full_cache_path = os.path.join(self.base_project_dir, 'data', 'cache', 'full_cache.json')
        full_specific_cache_path = os.path.join(self.base_project_dir, 'data', 'cache', 'full_specific_cache.json')
        full_page_id_cache_path = os.path.join(self.base_project_dir, 'data', 'cache', 'full_page_id_cache.json')
        entities_path = os.path.join(self.base_project_dir, 'data', 'embeddings', 'entities.jsonl')
        pages_path = os.path.join(self.base_project_dir, 'data', 'embeddings', 'page.csv')
        corpus_path = os.path.join(self.base_project_dir, self.input_directory)
        test_path = os.path.join(self.base_project_dir, self.output_directory, 'test.tsv')
        dev_path = os.path.join(self.base_project_dir, self.output_directory, 'dev.tsv')
        train_path = os.path.join(self.base_project_dir, self.output_directory, 'train.tsv')

        if not os.path.isfile(full_cache_path):
            mapper = ClassMapper.from_entities(entities_path)
            mapper.build_nkjp_cache_and_store(full_cache_path, full_specific_cache_path)

        if not os.path.isfile(full_page_id_cache_path):
            mapper = TitleMapper()
            mapper.load_entities(entities_path)
            mapper.load_pages(pages_path)
            mapper.build_cache_and_store(full_page_id_cache_path)
        self.entity_id_to_nkjp_class = ClassMapper.from_full_cache(full_cache_path)
        self.entity_id_to_nkjp_specific_class = ClassMapper.from_full_cache(full_specific_cache_path)
        self.page_id_to_wikidata_id = TitleMapper.from_full_cache(full_page_id_cache_path)
        test_file = open(test_path, 'w+', encoding='utf-8')
        dev_file = open(dev_path, 'w+', encoding='utf-8')
        train_file = open(train_path, 'w+', encoding='utf-8')
        start = time.time()
        generator = self.line_batch_generator(corpus_path)
        output_file_generator = self.target_set_generator({test_file: split_weights[0], dev_file: split_weights[1],
                                                           train_file: split_weights[2]})

        self.set_up()
        for line in generator:
            if line_limit is not None and line_limit <= line_counter:
                break
            line_counter += 1
            self.process_line(line)

            if len(self.articles) > 1000:
                self.filter_articles()

                for article in self.articles:
                    target_file = next(output_file_generator)
                    article.write_to_file(target_file)
                self.articles = set()

        print('total articles: %d' % len(self.articles))
        self.filter_articles()

        for article in self.articles:
            target_file = next(output_file_generator)
            article.write_to_file(target_file)
        print('total articles: %d' % len(self.articles))
        # print('too long sentences: %d' % len(list(filter(lambda a: not a.are_sentences_short_enough(), self.articles))))
        # print('missing ids: %d' % missing_ids)
        # print('wrong title spans: %d' % wrong_title_spans)
        print(time.time() - start)

