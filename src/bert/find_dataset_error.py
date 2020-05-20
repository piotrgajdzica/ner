import os

if __name__ == '__main__':

    base_dir = r'C:\Users\piotrek\Desktop\inf\magisterka\ner\data\training_datasets\wikipedia_filtered\bert\0.2'

    files = ['test', 'train', 'dev']


    last_line_empty = False
    counter = 0
    sentence_len = 0

    for line in open(os.path.join(base_dir, 'dev' + '.txt'), encoding='utf-8').readlines():
        counter += 1
        if line == '\n':
            assert sentence_len <= 128
            sentence_len = 0
            if last_line_empty:
                print(counter)
            last_line_empty = True
            continue
        else:
            last_line_empty = False
        try:
            token, tag = line.split(' ')
        except ValueError:
            print(counter)
            print(line)
        assert len(tag) < 15
        assert len(tag) > 0
        assert len(token) < 40
        assert len(token) > 0