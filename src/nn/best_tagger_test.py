from flair.data import Sentence
from flair.models import SequenceTagger

if __name__ == '__main__':

    tagger_path = r'C:\Users\piotrek\Desktop\inf\magisterka\ner\data\tagger\final-model.pt'

    tagger = SequenceTagger.load(tagger_path)
    sentence = Sentence('George Washington went to Washington .')

    # predict NER tags
    print(tagger.predict(sentence))
