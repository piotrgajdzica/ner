
import nltk.data
import requests

if __name__ == '__main__':
    # pickle = 'tokenizers/punkt/english.pickle'
    # try:
    #     tokenizer = nltk.data.load(pickle)
    # except LookupError:
    #     nltk.download('punkt')
    #     tokenizer = nltk.data.load(pickle)
    #
    # data = "Na terenie diecezji lubelsko-chełmskiej w 2019 funkcjonowały 53 świątynie[1][a], z czego dwadzieścia pięć to obiekty zabytkowe[2]. Najstarszą cerkwią należącą do diecezji (jak również najstarszą świątynią we władaniu Polskiego Autokefalicznego Kościoła Prawosławnego) jest cerkiew Zaśnięcia Najświętszej Maryi Panny w Szczebrzeszynie, wzniesiona w XVI w.[3]. Drugą co do wieku budowlą sakralną w diecezji lubelsko-chełmskiej jest natomiast katedralny sobór Przemienienia Pańskiego w Lublinie, wyświęcony w 1633[4]. Spośród pozostałych zabytkowych cerkwi dziewięć to budowle wzniesione jako świątynie unickie w XVIII i XIX w. (do 1875), zaś trzynaście – cerkwie prawosławne zbudowane po likwidacji unickiej diecezji chełmskiej w 1875, a przed wycofaniem się Rosjan z Królestwa Polskiego w 1915. Cerkiew św. Jana Teologa w Chełmie została wzniesiona jako prawosławna cerkiew wojskowa w latach 1846–1852[5]."
    # print('\n-----\n'.join(tokenizer.tokenize(data)))
    response = requests.post('http://192.168.99.100:9200', data="Ala ma kota. Kota ma Ala")
    tagged = response.content.decode('utf8')
    first_meaning = False
    chosen_tags = []
    for line in tagged.split('\n'):
        if line.startswith('\t'):
            if first_meaning:
                first_meaning = False
                chosen_tags[-1] += "\t" + (line[1:])
        else:
            first_meaning = True
            chosen_tags.append(line.split('\t')[0])
    print('\n'.join(chosen_tags))
