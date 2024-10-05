import classla

classla.download('sr')

nlp = classla.Pipeline('sr', processors='tokenize,pos,lemma')


def lemmatize(text):
    return " ".join(nlp(text).get('lemma'))




