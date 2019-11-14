import flair
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.data import Corpus
from flair.datasets import ColumnCorpus


if __name__ == '__main__':

    # define columns
    columns = {0: 'text', 1: 'lemma', 2: 'space', 3: 'morph', 4: 'ner'}

    # this is the folder in which train, test and dev files reside
    data_folder = '../data/tokens-with-entities-tags-and-classes_sample'

    # init a corpus using column format, data folder and the names of the train, dev and test files
    corpus: Corpus = ColumnCorpus(data_folder, columns,
                                  train_file='train.tsv',
                                  test_file='test.tsv',
                                  dev_file='dev.tsv')
    print(corpus.obtain_statistics())
    print(corpus)
    print(corpus.test[0].to_tagged_string('ner'))
