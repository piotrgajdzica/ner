import os

from preprocessing.data_filter import DataFilter

if __name__ == '__main__':

    base_dir = r'C:\Users\piotrek\Desktop\inf\magisterka\ner'


    architecture_map = {
        'flair': ('test.tsv', 'test.tsv', '\t'),
        'bert': ('bert/test.txt', 'test_predictions.txt', ' '),
        'xlmr': ('bert/test.txt', 'test_predictions.txt', ' '),
        'slavic': ('bert/test.txt', 'test_predictions.txt', ' '),
        'polish': ('bert/test.txt', 'test_predictions.txt', ' '),
    }
    for model in open('models').readlines():
        if model[0] != '#':
            model = model[:-1]
            print(model)
            dataset = model.split('-')[1]
            architecture = model.split('-')[2]

            test_file = os.path.join(base_dir, 'data', 'training_datasets', dataset, architecture_map[architecture][0])
            prediction_file = os.path.join(base_dir, 'final_models', 'test_results_unfiltered', model, architecture_map[architecture][1])
            output_file = os.path.join(base_dir, 'final_models', 'test_results_filtered', '%s.tsv' % model)

            test_lines = open(test_file, encoding='utf-8').readlines()
            prediction_lines = open(prediction_file, encoding='utf-8').readlines()

            test_generator = DataFilter.line_batch_generator(test_file)
            prediction_generator = DataFilter.line_batch_generator(prediction_file)

            output_batch = []

            try:
                os.unlink(output_file)
            except FileNotFoundError:
                pass
            output_file = open(output_file, mode='w', encoding='utf-8')

            counter = 0
            end = False
            while not end:
                try:
                    test_line = next(test_generator).split()
                    prediction_line = next(prediction_generator).split()
                    if not test_line:
                        output_batch.append('\n')
                    else:
                        token = test_line[0]
                        test_tag = test_line[-2] if architecture == 'flair' else test_line[-1]
                        predicted_tag = prediction_line[-2] if architecture == 'flair' else prediction_line[-1]
                        output_batch.append('%s\t%s\t%s\n' % (token, test_tag, predicted_tag))
                        if len(output_batch) > 100:
                            output_file.writelines(output_batch)
                            output_batch.clear()
                except StopIteration:
                    try:
                        next(prediction_generator)
                    except StopIteration:
                        end = True

            output_file.writelines(output_batch)
            # print(len(test_lines))
            # print(len(prediction_lines))
