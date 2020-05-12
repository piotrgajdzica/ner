import datetime
import os
import sys

from postprocessing.build_poleval_tests import build_tests
from postprocessing.poleval_ner_test import computeScores

if __name__ == '__main__':
    poleval_dir = r'C:\Users\piotrek\Desktop\inf\magisterka\ner\data\poleval_scripts\test'
    userfile = os.path.join(poleval_dir, 'tagged.json')
    goldfile = os.path.join(poleval_dir, 'gold.json')
    sys.stdout = open('reports/report%s.txt' % datetime.datetime.now().strftime("%Y-%m-%d,%H;%M;%S"), 'w')
    for line in open('models', encoding='utf-8'):
        model = line[:-1]
        if model and not model.startswith('#'):
            print()
            build_tests(poleval_dir, model)
            computeScores(model, goldfile, userfile, htype="split")
