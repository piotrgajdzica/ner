import os
import time

from preprocessing.class_mapper import ClassMapper

if __name__ == '__main__':

    full_cache_path = os.path.join('..', 'data', 'full_cache.json')
    entities_path = os.path.join('..', 'data', 'entities.jsonl')
    corpus_path = os.path.join('..', 'data', 'tokens-with-entities-and-tags.tsv')
    output_path = os.path.join('..', 'data', 'tokens-with-entities-tags-and-classes.tsv')

    if not os.path.isfile(full_cache_path):
        mapper = ClassMapper.from_entities(entities_path)
        mapper.build_nkjp_cache_and_store(full_cache_path)
    entity_id_to_nkjp_class = ClassMapper.from_full_cache(full_cache_path)
    f = open(corpus_path, encoding='utf-8')
    # output = open(output_path, 'w', encoding='utf-8')
    counter = 0
    start = time.time()
    while True:
        lines = f.readlines(10000000)
        counter += len(lines)
        print(counter)
        if len(lines) == 0:
            break
        for line in lines:
            columns = line[:-1].split('\t')
            if len(columns) == 7:
                nkjp_class = None
                article_no, word, base, space, tags, entity, entity_wikidata_id = columns
                if entity_wikidata_id != '_':
                    entity_wikidata_id = int(entity_wikidata_id[1:])
                    nkjp_class = entity_id_to_nkjp_class.get(entity_wikidata_id)
                    if nkjp_class is not None:
                        print(word, entity, nkjp_class)
                # output.write('%s\t%s\t%s\t%s\t%s\t%s\n' % (article_no, word, base, space, tags, nkjp_class or '_'))
            else:
                pass
                # output.write(line)
    print(time.time() - start)
