import requests

if __name__ == '__main__':
    response = requests.post('http://192.168.99.100:9003', data='Ala ma kota. Janek ma psa. Mój kocie'.encode('utf-8')).content.decode(encoding='utf-8')
    print(response)