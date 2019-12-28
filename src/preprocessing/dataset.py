import os
import time
from random import random
from typing import List

from src.util.class_mapper import ClassMapper
from src.util.title_mapper import TitleMapper


class Token:
    def __init__(self, token, lemma, space, tags, entity, entity_wikidata_id, nkjp_class=None, nkjp_specific_class=None, start_tag=None):
        self.entity_wikidata_id = entity_wikidata_id
        self.entity = entity
        self.tags = tags
        self.space = space
        self.lemma = lemma.replace(' ', '_').replace(' ', '_')
        self.token = token.replace(' ', '_').replace(' ', '_')
        self.start_tag = start_tag
        self.nkjp_class = nkjp_class
        self.specific_nkjp_class = nkjp_specific_class

    def write_to_file(self, file):
        file.write('%s\t%s\t%s\t%s\t%s_%s\t%s_%s\n' % (
            self.token, self.lemma, self.space, self.tags, self.start_tag or '', self.nkjp_class or '',
            self.start_tag or '', self.specific_nkjp_class or ''))

    def __str__(self):
        return self.token

    def __repr__(self):
        return self.token


class Sentence:
    def __init__(self, tokens=None):
        self.tokens: List[Token] = tokens or []

    def is_valid(self):
        return self.size() < 60

    def size(self):
        return len(self.tokens)

    def write_to_file(self, file):
        for token in self.tokens:
            token.write_to_file(file)
        file.write('\n')


class Article:
    def __init__(self, doc_id, sentences=None, sentence_limit=None):
        self.doc_id = doc_id
        self.sentences: List[Sentence] = sentences or []
        self.sentence_limit = sentence_limit
        self.title_annotation_error = False

    def is_valid(self):
        return self.are_sentences_short_enough() and not self.title_annotation_error

    def are_sentences_short_enough(self):
        return all(sentence.is_valid() for sentence in self.sentences)

    def add_next_sentence(self, sentence: Sentence):
        if self.sentence_limit is None or len(self.sentences) < self.sentence_limit:
            self.sentences.append(sentence)

    def write_to_file(self, file):
        for sentence in self.sentences:
            sentence.write_to_file(file)

    def annotate_title(self, title, nkjp_class, nkjp_specific_class):
        span = self.get_title_span(title)
        if span is None:
            self.title_annotation_error = True
            return

        start, length = span
        start_tag = 'B'
        for token in self.sentences[0].tokens[start: start+length]:
            token.nkjp_class = nkjp_class
            token.nkjp_specific_class = nkjp_specific_class
            token.start_tag = start_tag
            start_tag = 'I'

    def get_title_span(self, title):
        title_tokens = title.split()
        tokens = self.sentences[0].tokens
        try:
            title_end_index = next(i for i, token in enumerate(tokens) if token.token == '–')
            tokens = tokens[:title_end_index]
        except StopIteration:
            pass

        title_end_index = len(tokens) - len(title_tokens) + 1

        for start_comparison in range(title_end_index):
            success = True
            idx = 0
            for title_token in title_tokens:
                token = tokens[idx + start_comparison].token.lower()
                idx += 1
                if title_token.lower() != token:
                    if title_token.startswith('('):
                        return (start_comparison, idx)
                    success = None
            if success:
                return (start_comparison, idx)
        return None


# def prepare_dataset(article_limit=None, base_dir='/net/people/plgpgajdzica/scratch/ner/data', corpus_dir='', skip_title=True):
#     full_cache_path = os.path.join(base_dir, 'full_cache.json')
#     full_specific_cache_path = os.path.join(base_dir, 'full_specific_cache.json')
#     full_page_id_cache_path = os.path.join(base_dir, 'full_page_id_cache.json')
#     entities_path = os.path.join(base_dir, 'entities.jsonl')
#     pages_path = os.path.join(base_dir, 'page.csv')
#     corpus_path = os.path.join(base_dir, 'tokens-with-entities-and-tags.tsv')
#     test_path = os.path.join(base_dir, corpus_dir, 'test.tsv')
#     dev_path = os.path.join(base_dir, corpus_dir, 'dev.tsv')
#     train_path = os.path.join(base_dir, corpus_dir, 'train.tsv')
#
#     if not os.path.isfile(full_cache_path):
#         mapper = ClassMapper.from_entities(entities_path)
#         mapper.build_nkjp_cache_and_store(full_cache_path)
#
#     if not os.path.isfile(full_page_id_cache_path):
#         mapper = TitleMapper()
#         mapper.load_entities(entities_path)
#         mapper.load_pages(pages_path)
#         mapper.build_cache_and_store(full_page_id_cache_path)
#     entity_id_to_nkjp_class = ClassMapper.from_full_cache(full_cache_path)
#     entity_id_to_nkjp_specific_class = ClassMapper.from_full_cache(full_specific_cache_path)
#     page_id_to_wikidata_id = TitleMapper.from_full_cache(full_page_id_cache_path)
#     test_file = open(test_path, 'w', encoding='utf-8')
#     dev_file = open(dev_path, 'w', encoding='utf-8')
#     train_file = open(train_path, 'w', encoding='utf-8')
#     start = time.time()
#     generator = line_batch_generator(corpus_path)
#     output_file_generator = target_set_generator({test_file: 1, dev_file: 1, train_file: 3})
#
#     articles: List[Article] = []
#     article = None
#     sentence = None
#     last_entity = None
#     for line in generator:
#         columns = line[:-1].split('\t')
#         if len(columns) == 7:
#             article_no, token, lemma, space, tags, entity, entity_wikidata_id = columns
#
#             if article is None or article_no != article.doc_id:
#                 if article_limit is not None and len(articles) > article_limit:
#                     break
#                 article = Article(article_no, sentence_limit=3)
#                 articles.append(article)
#
#             if sentence is None:
#                 sentence = Sentence()
#                 article.add_next_sentence(sentence)
#             token = Token(token, lemma, space, tags, entity, entity_wikidata_id)
#             sentence.tokens.append(token)
#
#             if entity_wikidata_id != '_':
#                 entity_wikidata_id = int(entity_wikidata_id[1:])
#                 token.nkjp_class = entity_id_to_nkjp_class.get(entity_wikidata_id)
#                 if token.nkjp_class is not None:
#                     token.start_tag = 'B' if last_entity != entity else 'I'
#
#                 # if nkjp_class is not None:
#                 #     print(token, entity, nkjp_class)
#
#             last_entity = entity
#         elif len(columns) != 1:
#             print('Invalid number of columns: %d' % len(columns))
#             print(columns)
#         else:  # we reached a blank line - meaning the sentence is over
#             sentence = None
#
#     missing_ids = 0
#     wrong_title_spans = 0
#     for article in articles:
#         if article.is_valid():
#             wikidata_json = page_id_to_wikidata_id.get(int(article.doc_id), None)
#             if wikidata_json is not None and wikidata_json['id'] is not None:
#                 nkjp_class = entity_id_to_nkjp_class.get(wikidata_json['id'], None)
#                 nkjp_specific_class = entity_id_to_nkjp_specific_class.get(wikidata_json['id'], None)
#                 if nkjp_class is not None:
#                     article.annotate_title(wikidata_json['title'], nkjp_class, nkjp_specific_class)
#                     if article.title_annotation_error:
#                         wrong_title_spans += 1
#             else:
#                 article.title_annotation_error = True
#                 missing_ids += 1
#         if article.is_valid():
#             target_file = next(output_file_generator)
#             article.write_to_file(target_file)
#     print('total articles: %d' % len(articles))
#     print('valid articles: %d' % len(list(filter(lambda a: a.is_valid(), articles))))
#     print('too long sentences: %d' % len(list(filter(lambda a: not a.are_sentences_short_enough(), articles))))
#     print('missing ids: %d' % missing_ids)
#     print('wrong title spans: %d' % wrong_title_spans)
#     print(time.time() - start)
#
#
# if __name__ == '__main__':
#     prepare_dataset(article_limit=1000, base_dir='C:\\Users\\piotrek\\Desktop\\inf\\magisterka\\ner\\data\\', skip_title=False)
#
