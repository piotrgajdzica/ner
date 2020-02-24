import requests

if __name__ == '__main__':
    response = requests.post('http://192.168.99.100:9003?input_format=lines&output_format=conll', data='Ala, ma kota. Janek ma psa.\n\nJanek ma psa.\n\nMój kocie'.encode('utf-8')).content.decode(encoding='utf-8')
    print(response)