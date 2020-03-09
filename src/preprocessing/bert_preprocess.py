import sys

from transformers import AutoTokenizer


def preprocess(dataset, destination, model_name_or_path, max_len):

    subword_len_counter = 0

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    destination_file = open(destination, 'w', encoding='utf-8')

    with open(dataset, "r", encoding='utf-8') as f_p:
        for line in f_p:
            line = line.rstrip()

            if not line:
                destination_file.write(line + '\n')
                subword_len_counter = 0
                continue

            token = line.split()[0]

            current_subwords_len = len(tokenizer.tokenize(token))

            # Token contains strange control characters like \x96 or \x95
            # Just filter out the complete line
            if current_subwords_len == 0:
                continue

            if (subword_len_counter + current_subwords_len) > max_len:
                destination_file.write('\n%s\n' % line)
                subword_len_counter = 0
                continue

            subword_len_counter += current_subwords_len

            destination_file.write(line + '\n')
