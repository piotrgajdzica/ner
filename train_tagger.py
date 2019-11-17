from preprocessing.dataset import prepare_dataset
import flair
import torch
from flair.datasets import ColumnCorpus
from flair.embeddings import FlairEmbeddings, StackedEmbeddings, WordEmbeddings, BertEmbeddings, OneHotEmbeddings
from flair.models import SequenceTagger, LanguageModel
from flair.trainers import ModelTrainer
from flair.visual.training_curves import Plotter
import argparse

from typing import List



def load_language_model_non_strict(model_file):
    state = torch.load(str(model_file), map_location=flair.device)

    model = LanguageModel(
        state["dictionary"],
        state["is_forward_lm"],
        state["hidden_size"],
        state["nlayers"],
        state["embedding_size"],
        state["nout"],
        state["dropout"],
    )
    model.load_state_dict(state["state_dict"], strict=False)
    model.eval()
    model.to(flair.device)

    return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Train and test model for tagging aspects and features (or any corpus similar to NER)')
    parser.add_argument('tagger_dir', help="Directory to save the best tagger and training results")
    parser.add_argument('corpus_dir', help="Directory with train.txt, test.txt and dev.txt files in CONLL format"
                                           " with the train-validation-test split training corpus\n"
                                           "CONLL columns - 0: text, 1: lemma, 2: space, "
                                           "3: morph, 4: ner")
    parser.add_argument('--hidden-size', type=int, dest='hidden_size', default=256,
                        help='Network hidden layer size')
    parser.add_argument('--downsample', '-s', type=float, dest='downsample', default=1.0,
                        help='Downsample the corpus to a given fraction')
    parser.add_argument('--dropout', '-d', type=float, dest='dropout', default=.2,
                        help='Network dropout probability')
    parser.add_argument('--learning-rate', '-l', type=float, dest='lr', default=.7,
                        help='Network learning rate')
    parser.add_argument('--batch-size', '-b', type=int, dest='batch_size', default=16,
                        help='Single training batch size')
    parser.add_argument('--max_epochs', type=int, dest='batch_size', default=3,
                        help='Single training batch size')
    parser.add_argument('--forward-path', type=str, dest='forward_path',
                        default='../data/wiki+nkjp-small-f.pt',
                        help="Path to forward language model", action='store')
    parser.add_argument('--embeddings-path', '-e', type=str, dest='embeddings_path',
                        default='../data/word_embeddings',
                        help="Path to word level embeddings in gensim format", action='store')
    parser.add_argument('--backward-path', type=str, dest='backward_path',
                        default='../data/wiki+nkjp-small-b.pt',
                        help="Path to backward language model", action='store')
    parser.add_argument('--use-morph', '-m', dest='use_morph',
                        default=False, help="Use morphosyntactic tags in training", action='store_true')
    parser.add_argument('--morph-embedding-len', type=int, dest='morph_embedding_len',
                        default=32, help="Size of the embedding layer for morph tags", action='store')
    parser.add_argument('--use-lemma', '-x', dest='use_lemma',
                        default=False, help="Use token lemma", action='store_true')
    parser.add_argument('--prepare_dataset', dest='prepare_dataset',
                        default=False, help="Prepare dataset", action='store_true')
    parser.add_argument('--use-embedding', '-i', dest='use_embedding',
                        default=False, help="Embed tagged word in embedding space", action='store_true')
    parser.add_argument('--use-space', '-p', dest='use_space',
                        default=False, help="Use information about whitespace", action='store_true')
    args = parser.parse_args()

    if args.prepare_dataset:
        prepare_dataset(limit_size=100000000)

    # 1. get the corpus
    columns = {1: 'text', 3: 'space', 5: 'ne'}
    if args.use_morph:
        columns[4] = 'morph'
    if args.use_lemma:
        columns[2] = 'lemma'

    corpus = ColumnCorpus(args.corpus_dir, columns)

    if args.downsample != 1.0:
        corpus.downsample(args.downsample)

    # 2. what tag do we want to predict?
    tag_type = 'ne'

    # 3. make the tag dictionary from the corpus
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

    # 4. initialize embeddings
    embedding_types: List[FlairEmbeddings] = [
        FlairEmbeddings(load_language_model_non_strict(args.forward_path), chars_per_chunk=128),
        FlairEmbeddings(load_language_model_non_strict(args.backward_path), chars_per_chunk=128),
        #WordEmbeddings(args.embeddings_path)
    ]
    # TODO
    if args.use_lemma:
        embedding_types.append(WordEmbeddings(args.embeddings_path, field='lemma'))

    if args.use_morph:
        embedding_types.append(OneHotEmbeddings(corpus=corpus, field='morph', embedding_length=args.morph_embedding_len))

    if args.use_embedding:
        embedding_types.append(OneHotEmbeddings(corpus=corpus, embedding_length=32))

    if args.use_Space:
        embedding_types.append(OneHotEmbeddings(corpus=corpus, field='space', embedding_length=1))

    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

    # 5. initialize sequence tagger
    tagger: SequenceTagger = SequenceTagger(hidden_size=args.hidden_size,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=tag_type,
                                            use_crf=True,
                                            dropout=args.dropout,
                                            )

    # 6. initialize trainer
    trainer = ModelTrainer(tagger, corpus)

    # 7. start training
    trainer.train(
        args.tagger_dir,
        learning_rate=args.lr,
        mini_batch_size=args.batch_size,
        monitor_test=True,
        monitor_train=True,
        patience=5,
        anneal_factor=0.5,
        embeddings_storage_mode='gpu',
        max_epochs=args.max_epochs,
        # use_amp=True,
    )

    # 8. plot weight traces (optional)
    # plotter = Plotter()
    # plotter.plot_weights('resources/taggers/example-ner/weights.txt')
