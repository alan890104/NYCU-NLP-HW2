from lib2to3.pgen2 import token
from re import sub
from typing import List, Union
from regex import R
import spacy
from spacy import tokens
from spacy import displacy
import textacy

NLP = spacy.load("en_core_web_sm")

# https://blog.csdn.net/u010087338/article/details/121055591
OBJECT_DEPS = {"dobj", "attr", "dative", "oprd"}
SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "agent", "expl"}
CONJ_DEP = {"CCONJ", "CONJ"}  # and or but
VERB_POS = {"AUX", "VERB"}  # is has will should


def Test_textacy(doc: Union[spacy.tokens.doc.Doc, spacy.tokens.span.Span]):
    for text in list(textacy.extract.subject_verb_object_triples(doc)):
        for x in text:
            print(x)
        print()


def ToSVG(path, doc):
    svg = displacy.render(doc, style='dep')
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg)


def SplitBySent(doc: tokens.doc.Doc) -> List[tokens.doc.Doc]:
    '''
    將一段文本以一句為單位進行分割

    Example
    -------
    I am Alan.I am handsome.
    1. I am Alan.
    2. I am handsome.
    '''
    sentence = []
    for token in doc.sents:
        sentence.append(token)
    return sentence


def SplitByCCONJ(doc: tokens.doc.Doc) -> List[tokens.doc.Doc]:
    '''
    將一句話以連接詞做分割, 分割成短句

    Example
    -------
    I am Alan and I am handsome.
    1. I am Alan.
    2. I am handsome.
    '''
    previous = 0
    sentence = []
    for token in doc:
        if token.dep_ == "cc" and token.pos_ in CONJ_DEP:
            text = doc[previous:token.i]
            sentence.append(text)
            previous = token.i+1
    if previous < len(doc):
        sentence.append(doc[previous:])
    return sentence


def FindSubject(doc: tokens.doc.Doc):
    subs = []
    for token in doc:
        if token.dep_ in SUBJECT_DEPS:
            subs.append(token.text)
    return ' '.join(subs)


def FindObject(doc: tokens.doc.Doc):
    '''
    取得物件, 除了OBJ以外, 也考慮是tag是WDT(ex: which goods)
    '''
    objs = []
    for token in doc:
        if token.dep_ in OBJECT_DEPS or token.tag_ == "WDT":
            objs.append(token.text)
    return ' '.join(objs)


def FindObjfromVerb(v: tokens.Token):
    objs = []
    for child in v.children:
        if child.dep_ in OBJECT_DEPS:
            objs.append(child.text)
    return objs

def findSubjectfromVerb(tokens: List[tokens.Token]):
    subs = []
    for token in tokens:
        if token.dep_ in SUBJECT_DEPS:
            # who which ...
            if token.tag_ == "WP": print(list(token.lefts))
            subs.append(token.text)
    return ' '.join(subs)


def findObjectfromVerb(tokens: List[tokens.Token]):
    objs = []
    for token in tokens:
        if token.dep_ in OBJECT_DEPS:
            objs.append(token.text)
    return ' '.join(objs)

def FindVerbs(sentence: tokens.doc.Doc) -> str:
    '''
    從短句中取得所有動詞
    '''
    svos = []
    verbs = [x for x in sentence if x.pos_ in VERB_POS]
    for v in verbs:
        subs = findSubjectfromVerb(list(v.lefts))
        objs = findObjectfromVerb(list(v.rights))
        if len(subs)==0 and len(objs)==0:
            continue
        svos.append((subs,v,objs))
    return svos


def FindSVO(doc: tokens.doc.Doc):
    sents = SplitBySent(doc)
    for sent in sents:
        sentence = SplitByCCONJ(sent)
        for s in sentence:
            print(s)
            svos = FindVerbs(s)
            for svo in svos:
                print(svo[0], "|", svo[1], "|", svo[2])
            print()



if __name__ == "__main__":
    # https://spacy.io/usage/linguistic-features

    # 分割成兩句話
    # doc = NLP("I 've been here 19 years and it 's never been a good time ")

    # verb and conj
    doc = NLP("He beat and hurt me.")

    # 比對的時候不管n't never等否定詞
    # doc = NLP("It did n't specify which goods in either case ")

    # 從動詞開始找主詞和受詞
    # doc = NLP("It 's an effective , maybe even subversive , way to get the message across .")

    # pass 
    # doc = NLP("She began her career as a chunky woman , awkward on stage and unsure of her musical direction .")

    # who(dep="")應該被表示成原本修飾的主格
    # doc  = NLP("A Spanish official , who had just finished a siesta and seemed not the least bit tense , offered what he believed to be a perfectly reasonable explanation for why the portable facilities were n't in service .")
    
    # doc  = NLP("The teacher found a book which the student lost.")

    # https://github.com/rock3125/enhanced-subject-verb-object-extraction/blob/ff8e7e5011c989d946579b537ccfe4191f8e2bfc/subject_verb_object_extract.py#L66
    # doc = NLP("A Spanish official , who had just finished a siesta and seemed not the least bit tense , offered what he believed to be a perfectly reasonable explanation for why the portable facilities were n't in service .")
    # doc = NLP( "It 's an effective , maybe even subversive , way to get the message across .")
    # doc = NLP("The girl kisses and hugs me.")
    # doc = NLP("Alan and Bob and Mike likes that girl.")
    # doc = NLP(" have been watching Namin for a year.")
    # doc = NLP("Sue asked George to respond her offer.")
    # doc = NLP("I love Namin and her sister and her brother.")
    # doc = NLP("The students and Alan thought that they had met us.")
    # doc = NLP("Christopher's excuse was that he had forgotten to set his alarm.")
    
    print('%10s %5s %10s %10s %10s %10s %10s' % ("word", "tag", "dep", "pos", "head", "left","right"))
    print('=================================================================================================')
    for word in doc:
        print('%10s %5s %10s %10s %10s %10s %10s' %
              (word.text, word.tag_, word.dep_, word.pos_, word.head.text, list(word.lefts), list(word.rights)))
    # FindSVO(doc)
    # displacy.serve(doc)