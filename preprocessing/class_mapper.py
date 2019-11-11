import json
import time


class ClassMapper:
    def __init__(self, id_to_classes, id_to_derived_classes):

        self.id_to_derived_classes = id_to_derived_classes
        self.id_to_classes = id_to_classes
        self.nkjp_cache = {}
        self.nkjp_cache = {
            5: 'persName',  # Q5 - human
            16334295: 'persName',  # Q16334295 - group of humans
            215627: 'persName',  # Q215627 - person
            95074: 'persName',  # Q95074 - fictional character
            43229: 'orgName',  # Q43229 - organization
            486972: 'settlement',  # Q486972 - human settlement
            6256: 'country',  # Q6256 - country
            245065: 'bloc',  # Q245065 - intergovernmental organization
            4286337: 'district',  # Q4286337 - city district
            1799794: 'region',  # Q1799794 - administrative territorial entity of a specific level
            811979: 'geogName',  # Q811979 - architectural structure
            15642541: 'geogName',  # Q15642541 - human-geographic territorial entity
            618123: 'geogName',  # Q618123 - geographical object
            2221906: 'geogName',  # Q2221906 - geographical location
        }

        self.class_priority = ['country', 'bloc', 'settlement', 'district', 'region', 'persName', 'geogrName',
                               'orgName']

    def choose_best_candidate(self, candidates):
        candidates = [candidate for candidate in candidates if candidate is not None]
        if len(candidates) == 0:
            return None
        else:
            for cl in self.class_priority:
                if cl in candidates:
                    return cl
        return candidates[0]

    def get_nkjp_class_for_wikidata_class(self, class_id, recursion_depth = 0):
        if recursion_depth > 100:
            return None
        if class_id in self.nkjp_cache:
            return self.nkjp_cache[class_id]

        candidates = [self.get_nkjp_class_for_wikidata_class(class_class_id, recursion_depth + 1)
                      for class_class_id in self.id_to_derived_classes.get(class_id, [])]
        nkjp_class = self.choose_best_candidate(candidates)
        self.nkjp_cache[class_id] = nkjp_class
        return nkjp_class

    def get_nkjp_class_for_wikidata_item(self, item_id):

        candidates = [self.get_nkjp_class_for_wikidata_class(item_class) for item_class in self.id_to_classes[item_id]]
        return self.choose_best_candidate(candidates)

    @classmethod
    def from_entities(cls, filename):
        f = open(filename, encoding='utf-8')

        id_to_classes = {}
        id_to_derived_classes = {}
        start = time.time()
        while True:
            lines = f.readlines(10000)

            line_count = len(lines)
            lines = '[%s]' % ','.join(lines)
            items = json.loads(lines)
            id_to_classes.update({int(item['id'][1:]):  [int(s[1:]) for s in item.get('P31', [])] for item in items})
            id_to_derived_classes.update(
                {int(item['id'][1:]): [int(s[1:]) for s in item.get('P279', [])] for item in items})
            if line_count == 0:
                break
        print(time.time() - start)
        return cls(id_to_classes, id_to_derived_classes)

    @classmethod
    def from_full_cache(cls, filename) -> dict:
        f = open(filename, encoding='utf-8')
        cache = json.loads(f.read())
        return {int(key): value for key, value in cache.items()}

    def build_nkjp_cache_and_store(self, filename):
        start = time.time()
        full_mapping = {}
        for item_id in self.id_to_classes:
            nkjp_class = self.get_nkjp_class_for_wikidata_item(item_id)
            if nkjp_class is not None:
                full_mapping[item_id] = nkjp_class
        open(filename, 'w', encoding='utf-8').write(json.dumps(full_mapping))
        print(time.time() - start)
