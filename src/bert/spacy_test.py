import spacy

if __name__ == '__main__':

    nlp = spacy.load('xx_ent_wiki_sm')
    doc = nlp(u"Jerzy Waszyngton, prezydent Stanów Zjednoczonych Ameryki pojechał do Waszyngtonu na obrady Departamentu "
              u"Skarbu odbywające się w Kapitolu.")

    for entity in doc.ents:
        print(entity.text, entity.label_, entity.start_char, entity.end_char)
