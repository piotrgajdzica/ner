from transformers import pipeline, AutoTokenizer, AutoModel

if __name__ == '__main__':

    pytorch_model = r'C:\Users\piotrek\Desktop\inf\magisterka\ner\data\embeddings\bert\slavic\pytorch_model.bin'
    config = r'C:\Users\piotrek\Desktop\inf\magisterka\ner\data\embeddings\bert\slavic\config.json'

    # tokenizer = AutoTokenizer.from_pretrained("djstrong/bg_cs_pl_ru_cased_L-12_H-768_A-12", config=config)

    # model = AutoModel.from_pretrained("djstrong/bg_cs_pl_ru_cased_L-12_H-768_A-12", config=config)


    # Allocate a pipeline for sentiment-analysis
    nlp = pipeline('ner', model='djstrong/bg_cs_pl_ru_cased_L-12_H-768_A-12', tokenizer='bert-base-cased', config=config)
    # nlp = pipeline('ner', model=model, tokenizer=tokenizer, config=config)
    print(nlp('Jerzy Waszyngton pojechał do Waszyngtonu'))

