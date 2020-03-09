from deeppavlov import configs, build_model

ner_model = build_model(configs.ner.ner_ontonotes_bert_mult, download=True)
print(ner_model([u"Jerzy Waszyngton, prezydent Stanów Zjednoczonych "
                 u"Ameryki pojechał do Waszyngtonu na obrady "
                 u"Departamentu Skarbu odbywające się w Kapitolu."
                 ]))
