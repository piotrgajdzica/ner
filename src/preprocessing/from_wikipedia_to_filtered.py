import regex

from src.preprocessing.data_filter import DataFilter
from src.preprocessing.dataset import Article, Sentence, Token


class AtLeastOneTagSentence(Sentence):

    def should_skip(self) -> bool:
        pattern = regex.compile(r"^\p{Lu}")
        useful_tokens = 0

        for token in self.tokens[1:]:
            if token.entity_wikidata_id != '_':
                if token.nkjp_class is not None:
                    useful_tokens += 1
            elif pattern.match(token.token):
                return True
        return useful_tokens == 0

    def write_to_file(self, file):
        if self.should_skip():
            return
        for token in self.tokens:
            token.write_to_file(file)
        file.write('\n')


class AtLeastOneTagDataFilter(DataFilter):

    def __init__(self, *args, **kwargs):
        self.article = None
        self.sentence = None
        self.last_entity = None
        self.total_sentence_count = 0
        # self.total_sentence_limit = total_sentence_limit
        super().__init__(*args, **kwargs)

    def filter_articles(self):
        for article in self.articles.copy():
            if not article.is_valid():
                self.articles.remove(article)

    def set_up(self):
        pass

    def process_line(self, line: str):
        # if self.total_sentence_limit is not None and self.total_sentence_limit <= self.total_sentence_count:
        #     return
        columns = line[:-1].split('\t')
        if len(columns) == 7:
            article_no, token, lemma, space, tags, entity, entity_wikidata_id = columns

            if self.article is None or article_no != self.article.doc_id:
                if self.article is not None:
                    self.articles.add(self.article)
                self.article = Article(article_no)

            if self.sentence is None:
                self.sentence = AtLeastOneTagSentence()
                self.article.add_next_sentence(self.sentence)
            token = Token(token, lemma, space, tags, entity, entity_wikidata_id)
            self.sentence.tokens.append(token)

            if entity_wikidata_id != '_':
                entity_wikidata_id = int(entity_wikidata_id[1:])
                token.nkjp_class = self.entity_id_to_nkjp_class.get(entity_wikidata_id)
                token.specific_nkjp_class = self.entity_id_to_nkjp_specific_class.get(entity_wikidata_id)
                if token.nkjp_class is not None:
                    token.start_tag = 'B' if self.last_entity != entity else 'I'

            self.last_entity = entity
        elif len(columns) != 1:
            print('Invalid number of columns: %d' % len(columns))
            print(columns)
        else:  # we reached a blank line - meaning the sentence is over
            self.sentence = None


def process(base_dir):
    AtLeastOneTagDataFilter(
        'data/unfiltered_datasets/poleval',
        'data/training_datasets/wikipedia_filtered',
        base_dir)\
        .filter_data_and_save()


if __name__ == '__main__':
    process(r'C:\Users\piotrek\Desktop\inf\magisterka\ner')
