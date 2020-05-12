import json
import os
import random


def build_tests(test_dir, model):

    model += '.tsv'
    model_dir = r'C:\Users\piotrek\Desktop\inf\magisterka\ner\final_models\test_results_filtered'

    fileContent = []
    gold_answers = []
    tagged_answers = []

    line_number = 0
    current_gold_tag = 'O'
    gold_tag_start = None
    gold_text_buffer = []
    current_tag = 'O'
    tag_start = None
    text_buffer = []

    with open(os.path.join(model_dir, model), encoding='utf-8') as model_file:
        for line in model_file.readlines():
            if len(line) > 1:
                token, gold, tag = line[:-1].split('\t')
                fileContent.append(token)
                if gold == 'O' or gold.startswith('B'):
                    if current_gold_tag != 'O':
                        gold_answers.append('\t%s %d %d\t%s\n' %
                                            (current_gold_tag.split('-')[1], gold_tag_start, line_number,
                                             ''.join(gold_text_buffer)))

                if gold.startswith('B'):
                    gold_tag_start = line_number
                    gold_text_buffer.clear()
                gold_text_buffer.append(token)
                current_gold_tag = gold

                if tag == 'O' or tag.startswith('B'):
                    if current_tag != 'O':
                        tagged_answers.append('%s %d %d\t%s\n' %
                                            (current_tag.split('-')[1], tag_start, line_number, ''.join(text_buffer)))

                if tag.startswith('B'):
                    tag_start = line_number
                    text_buffer.clear()
                text_buffer.append(token)
                current_tag = tag

                line_number += len(token)
            else:
                pass
    gold_json = {
        "questions": [
            {
                "input": {
                    "fileContent": ''.join(fileContent),
                    "fname": "id"
                },
                "answers": [
                    ''.join(gold_answers)
                ]
            }
        ]
    }

    tagged_json = [{
        "text":''.join(fileContent),
        "id":"id",
        "answers": ''.join(tagged_answers)
    }]

    userfile = os.path.join(test_dir, 'tagged.json')
    goldfile = os.path.join(test_dir, 'gold.json')
    open(userfile, mode='w', encoding='utf-8').write(json.dumps(tagged_json, indent=4))
    open(goldfile, mode='w', encoding='utf-8').write(json.dumps(gold_json, indent=4))
