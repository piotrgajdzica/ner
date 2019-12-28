from src.preprocessing.data_filter import DataFilter
from src.preprocessing.dataset import Article, Sentence, Token


class ThreeSentenceDataFilter(DataFilter):

    def __init__(self, total_sentence_limit=None, *args, **kwargs):
        self.article = None
        self.sentence = None
        self.last_entity = None
        self.total_sentence_count = 0
        self.total_sentence_limit = total_sentence_limit
        super().__init__(*args, **kwargs)

    def filter_articles(self):
        missing_ids = 0
        wrong_title_spans = 0
        for article in self.articles.copy():
            if article.is_valid():
                wikidata_json = self.page_id_to_wikidata_id.get(int(article.doc_id), None)
                if wikidata_json is not None and wikidata_json['id'] is not None:
                    nkjp_class = self.entity_id_to_nkjp_class.get(wikidata_json['id'], None)
                    nkjp_specific_class = self.entity_id_to_nkjp_class.get(wikidata_json['id'], None)
                    if nkjp_class is not None:
                        article.annotate_title(wikidata_json['title'], nkjp_class, nkjp_specific_class)
                        if article.title_annotation_error:
                            wrong_title_spans += 1
                else:
                    article.title_annotation_error = True
                    missing_ids += 1
            if not article.is_valid():
                self.articles.remove(article)

    def set_up(self):
        pass

    def process_line(self, line: str):
        if self.total_sentence_limit is not None and self.total_sentence_limit <= self.total_sentence_count:
            return
        columns = line[:-1].split('\t')
        if len(columns) == 7:
            article_no, token, lemma, space, tags, entity, entity_wikidata_id = columns

            if self.article is None or article_no != self.article.doc_id:

                if self.article is not None:
                    self.articles.add(self.article)
                self.article = Article(article_no, sentence_limit=3)
                self.total_sentence_count += 3

            if self.sentence is None:
                self.sentence = Sentence()
                self.article.add_next_sentence(self.sentence)
            token = Token(token, lemma, space, tags, entity, entity_wikidata_id)
            self.sentence.tokens.append(token)

            if entity_wikidata_id != '_':
                entity_wikidata_id = int(entity_wikidata_id[1:])
                token.nkjp_class = self.entity_id_to_nkjp_class.get(entity_wikidata_id)
                token.specific_nkjp_class = self.entity_id_to_nkjp_specific_class.get(entity_wikidata_id)
                if token.nkjp_class is not None:
                    token.start_tag = 'B' if self.last_entity != entity else 'I'

                # if nkjp_class is not None:
                #     print(token, entity, nkjp_class)

            self.last_entity = entity
        elif len(columns) != 1:
            print('Invalid number of columns: %d' % len(columns))
            print(columns)
        else:  # we reached a blank line - meaning the sentence is over
            self.sentence = None


def process(base_dir):
    ThreeSentenceDataFilter(
        None,
        'data/unfiltered_datasets/poleval',
        'data/training_datasets/wikipedia_three_sentences',
        base_dir)\
        .filter_data_and_save()


if __name__ == '__main__':
    process(r'C:\Users\piotrek\Desktop\inf\magisterka\ner')
