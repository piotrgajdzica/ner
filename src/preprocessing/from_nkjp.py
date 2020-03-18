import os

from src.preprocessing.data_filter import DataFilter

if __name__ == '__main__':
    base_dir = r'C:\Users\piotrek\Desktop\inf\magisterka\ner'
    input_dir = os.path.join(base_dir, 'data/unfiltered_datasets/conll')
    morph_dir = os.path.join(base_dir, 'data/unfiltered_datasets/conll_morph')
    # file = 'wiki.dsmb.3886000-3953000.txt'
    annotated_tokens = []
    tagged_tokens = []
    full_page_id_cache_path = os.path.join(base_dir, 'data/cache/full_page_id_cache.json')

    corpus_output_dir = os.path.join(base_dir, 'data/training_datasets/nkjp')
    test_path = os.path.join(base_dir, corpus_output_dir, 'test.tsv')
    dev_path = os.path.join(base_dir, corpus_output_dir, 'dev.tsv')
    train_path = os.path.join(base_dir, corpus_output_dir, 'train.tsv')
    test_file = open(test_path, 'w', encoding='utf-8')
    dev_file = open(dev_path, 'w', encoding='utf-8')
    train_file = open(train_path, 'w', encoding='utf-8')
    output_file_generator = DataFilter.target_set_generator({test_file: 1, dev_file: 1, train_file: 8})
    f = next(output_file_generator)

    for input_file_path in os.listdir(input_dir):
        f = next(output_file_generator)
        input_file = open(os.path.join(input_dir, input_file_path), encoding='utf-8')
        input_file_morph = open(os.path.join(morph_dir, input_file_path), encoding='utf-8')

        input_lines = input_file.readlines()
        input_morph = input_file_morph.readlines()

        for line, line_morph in zip(input_lines, input_morph):
            if len(line) > 1:

                line = line.replace('\n', '')
                line_morph = line_morph.replace('\n', '')
                # print(line)
                # print(line.split('\t'))
                # print(line[:-2].split('\t'))
                try:
                    token, lemma, space, tag, tag2, _ = line.split('\t')
                    token_morph, lemma_morph, space_morph, tag_morph, _ = line_morph.split('\t')
                except ValueError:
                    print(line)
                    continue
                assert token == token_morph
                assert lemma == lemma_morph
                assert space == space_morph
                token = token.replace(' ', '_').replace(' ', '_')
                lemma = lemma.replace(' ', '_').replace(' ', '_')
                assert len(tag2) > 0

                if tag in ['B-date', 'I-date', 'B-time', 'I-time'] or tag == 'O':
                    tag = 'O'
                    tag2 = 'O'

                f.write('%s\t%s\t%s\t%s\t%s\t%s\n' % (token, lemma, space, tag_morph, tag, tag2))
            else:
                f.write(line)
