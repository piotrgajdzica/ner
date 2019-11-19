import json
import time


class TitleMapper:
    def __init__(self):
        self.title_to_wikidata_id = {}
        self.page_id_to_title = {}
        self.page_id_to_wikidata_id = {}

    def load_pages(self, file):
        for line in open(file, encoding='utf-8').readlines():
            page_id, title, *_ = line.split(',')
            page_id = int(page_id)
            if page_id in self.page_id_to_title:
                print('Error: duplicate key in pages: %s,%s', page_id, title)
            self.page_id_to_title[page_id] = title

    def load_entities(self, file):
        for line in open(file, encoding='utf-8').readlines():
            j = json.loads(line)
            title = j['wiki']['pl']
            wikidata_id = int(j['id'][1:])
            if title is not None:
                if title in self.title_to_wikidata_id:
                    print('Error: duplicate key in entities: %s,%s', title, wikidata_id)
                self.title_to_wikidata_id[title] = wikidata_id

    def get_wikidata_id_by_page_id(self, page_id):
        if page_id not in self.page_id_to_wikidata_id:
            self.page_id_to_wikidata_id[page_id] = self.title_to_wikidata_id[self.page_id_to_title[page_id]]
        return self.page_id_to_wikidata_id[page_id]

    @classmethod
    def from_full_cache(cls, filename) -> dict:
        f = open(filename, encoding='utf-8')
        cache = json.loads(f.read())
        return {int(key): value for key, value in cache.items()}

    def build_cache_and_store(self, filename):
        start = time.time()

        for page_id in self.page_id_to_title:
            title = self.page_id_to_title[page_id]
            wikidata_id = self.title_to_wikidata_id.get(title, None)
            if wikidata_id is not None:
                self.page_id_to_wikidata_id[page_id] = {'id': wikidata_id, 'title': title}
            else:
                self.page_id_to_wikidata_id[page_id] = {'id': None, 'title': title}

        open(filename, 'w', encoding='utf-8').write(json.dumps(self.page_id_to_wikidata_id))
        print(time.time() - start)
