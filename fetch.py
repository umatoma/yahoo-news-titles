from time import sleep
from urllib import request
from bs4 import BeautifulSoup

categories = [
    'domestic',
    'world',
    'economy',
    'entertainment',
    'sports',
    'computer',
    'science',
    'local'
]
for category in categories:
    print('category:', category)
    with open('%s.txt' % category, 'w') as f:
        for page in range(50):
            html = request.urlopen('https://news.yahoo.co.jp/list/?c=%s&p=%s' % (category, page)).read()
            soup = BeautifulSoup(html, 'html.parser')
            for ttl in soup.select('.listArea .list li .ttl'):
                f.write(ttl.string + '\n')
                print('ttl:', ttl.string)
            sleep(1)
