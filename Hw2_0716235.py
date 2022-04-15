# Author: Yu-Lun Hsu
# Student ID: 0716235
# HW ID: hw2
# Due Date: 04/16/2022


import re
from typing import Counter, List, Tuple

import pandas as pd
import spacy
from spacy import displacy
from spacy.tokens import Token
from spacy.tokens.doc import Doc

SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"}
OBJECT_DEPS = {"dobj", "dative", "attr", "oprd", "pobj"}
NOUN_POS = {"NOUN", "PRON", "PROPN"}

NLP = spacy.load("en_core_web_trf")


def show_tree(doc: Doc):
    '''
    Show word ,tag, dep, pos, ent, head, left, right of given doc
    '''
    print('%10s %5s %10s %10s %10s %10s %10s %10s' %
          ("word", "tag", "dep", "pos", "entity", "head", "left", "right"))
    print('=================================================================================================')
    for word in doc:
        print('%10s %5s %10s %10s %10s %10s %10s %10s' %
              (word.text, word.tag_, word.dep_, word.pos_, word.ent_type_, word.head.text, list(word.lefts), list(word.rights)))


def read_CSV(path: str) -> pd.DataFrame:
    '''
    read csv file from given path
    '''
    assert path.find(".csv") != -1, "expected to be csv file"
    df = pd.read_csv(path)
    return df


def removeSign(x: str):
    '''
    Remove (` , '',  \") from the original text.
    '''
    x1 = re.sub("`+|'{2}|\"", " ", x)
    x2 = re.sub(' +', ' ', x1)
    return x2


def lower(x: str):
    '''
    Transform letter to lower
    '''
    return x.lower()


def preprocessCSV(df: pd.DataFrame):
    '''
    Replace bad sign
    '''
    df["S"] = df["S"].apply(removeSign).apply(lower)
    df["V"] = df["V"].apply(removeSign).apply(lower)
    df["O"] = df["O"].apply(removeSign).apply(lower)
    df["sentence"] = df["sentence"].apply(removeSign).apply(lower)
    return df


def include_noun(collected: List[Token]) -> bool:
    '''
    Check if token array include noun
    '''
    for c in collected:
        if c.pos_ in {"NOUN", "PRON", "PROPN"}:
            return True
    return False


def is_verb(token: Token) -> bool:
    '''
    determine whether a token is a verb (ignore AUX with their head is VERB)
    ex: "I have been watching Namin for a year." =>  return (watching, )
    '''
    if token.pos_ == "VERB":
        return True
    if token.pos_ == "AUX" and token.head.pos_ != "VERB":
        return True
    return False


def include_verb(collected: List[Token]) -> bool:
    '''
    Check if token array include verb
    '''
    for c in collected:
        if is_verb(c):
            return True
    return False


def ExtractDocIndexArr(doc: Doc, S: str, V: str, O: str) -> Tuple[bool, List[Token], List[Token], List[Token]]:
    '''
    Return the location of given text in doc.

    Rules
    -----------
        1. Subject and Object can not include any verb, and should contain at least one {"NOUN", "PRON", "PROPN"}
        2. Verb cannot inlucde any noun, and should contain at least one {VERB, AUX}

    if success:
        return (True,S,V,O)
    else:
        return (False, [],[],[])
    '''
    i, j = 0, 0

    collected_S = []
    arr_S = S.split()
    while i != len(doc) and j != len(arr_S):
        if doc[i].text == arr_S[j]:
            if (doc[i].dep_ in SUBJECT_DEPS or doc[i].pos_ in NOUN_POS) and doc[i].tag_ not in {"PRP$"} and doc[i].ent_type_ not in {"TIME", "DATE"}:
                collected_S.append(doc[i])
            i += 1
            j += 1
        else:
            collected_S = []
            i += 1
            j = 0

    if len(collected_S) == 0:
        print("Subject collected S empty")
        return False, [], [], []

    if include_verb(collected_S):
        print("Subject cannot include verb")
        return False, [], [], []

    if not include_noun(collected_S):
        print("Subject not found noun")
        return False, [], [], []

    j = 0
    collected_V = []
    arr_V = V.split()
    while i != len(doc) and j != len(arr_V):
        if doc[i].text == arr_V[j]:
            collected_V.append(doc[i])
            i += 1
            j += 1
        else:
            collected_V = []
            i += 1
            j = 0

    if len(collected_V) == 0:
        print("Verb collected V empty")
        return False, [], [], []

    if include_noun(collected_V):
        print("Verb cannot include noun")
        return False, [], [], []

    if not include_verb(collected_V):
        print("Verb not found verb")
        return False, [], [], []

    j = 0
    collected_O = []
    arr_O = O.split()
    while i != len(doc) and j != len(arr_O):
        if doc[i].text == arr_O[j]:
            if (doc[i].dep_ in OBJECT_DEPS or doc[i].pos_ in NOUN_POS)\
                    and doc[i].tag_ not in {"PRP$"}\
                    and doc[i].ent_type_ not in {"TIME", "DATE"}:
                collected_O.append(doc[i])
            i += 1
            j += 1
        else:
            collected_O = []
            i += 1
            j = 0

    if len(collected_O) == 0:
        print("Object collected O empty")
        return False, [], [], []

    if include_verb(collected_O):
        print("Object cannot include verb")
        return False, [], [], []

    if not include_noun(collected_O):
        print("Object not found noun")
        return False, [], [], []

    return True, collected_S, collected_V, collected_O


def subj_in_conj_check(noun: Token, subs: List[Token]) -> bool:
    exist = False
    for s in subs:
        rights = list(s.rights)
        deps = {t.dep_ for t in rights}
        if "cc" in deps:
            subjects = [
                t for t in rights if t.dep_ in SUBJECT_DEPS or t.dep_ == "conj"]
            exist = (noun in subjects) or subj_in_conj_check(noun, subjects)
            if exist:
                return True
    return False


def subj_in_ancestors(noun: Token, v: Token) -> bool:
    current = v.head
    while current.pos_ != "VERB" and current.pos_ not in NOUN_POS and current.head != current:
        current = current.head
    if current.pos_ == "VERB":
        subs = [t for t in current.lefts]
        if noun in subs:
            return True
        elif subj_in_conj_check(noun, subs):
            return True
        elif current.head != current:
            return subj_in_ancestors(noun, current)
    elif current.pos_ in NOUN_POS:
        return noun == current


def SubjectCheck(noun: Token, v: Token) -> bool:
    lefts = [x for x in v.lefts if x.dep_ in SUBJECT_DEPS]
    # Find subjects in direct left children (most common case)
    if noun in lefts:
        print("Found subject in direct children.")
        return True
    # Find subjects in conjunction of left children
    elif len(lefts) > 0 and subj_in_conj_check(noun, lefts):
        print("Found subject in conjunction.")
        return True
    # Find implicit subject from ancesters
    if subj_in_ancestors(noun, v):
        print("Found subject in ancestors")
        return True
    return False


def is_passive_verb(v: Token) -> bool:
    '''
    Check if a verb is passive, find auxpass dep_ from its left child
    '''
    for child in v.lefts:
        if child.dep_ == "auxpass":
            return True
    return False


def obj_in_continuos_verb(noun: Token, v: Token) -> bool:
    '''
    Check if a verb's right side is also a verb in to+V or v+Ving form

    Example
    ----------
    I am considering selling the house.
    for verb "considering", 
    we have to find obj in the verb selling 

    They planned to make a cake for their mother.
    for verb "plan"
    we have to find obj in "make"

    She kissed and hugged me.
    for verb kissed
    we have to find obj in hugged
    '''
    rights = list(v.rights)
    #  To + V or V + Ving
    for child in rights:
        if is_verb(child) and child.dep_ in {"xcomp", "ccomp"}:
            if ObjectCheck(noun, child):
                return True

    # VERB + CCONJ + VERB
    if len(rights) > 1 and rights[0].pos_ == 'CCONJ':
        for r in rights[1:]:
            if is_verb(r):
                if ObjectCheck(noun, r):
                    return True
    return False


def obj_in_conj_check(noun: Token, objs: List[Token]) -> bool:
    '''
    Check conjunction objects joined by "cc"
    '''
    for o in objs:
        rights = list(o.rights)
        deps = {t.dep_ for t in rights}
        if "cc" in deps:
            objects = [
                t for t in rights if t.dep_ in OBJECT_DEPS or t.dep_ == "conj"]
            if noun in objects:
                return True
            if obj_in_conj_check(noun, objects):
                return True
    return False


def obj_in_preposition_check(noun: Token, tokens: List[Token]) -> bool:
    for t in tokens:
        if t.pos_ == "ADP" and t.dep_ in {"prep", "agent", "dative"}:
            objs = [r for r in t.rights if r.dep_ in OBJECT_DEPS or r.pos_ == "PRON"]
            if noun in objs:
                return True
            if obj_in_conj_check(noun, objs):
                return True
    return False


def ObjectCheck(noun: Token, v: Token) -> bool:
    '''
    Example
    ---------
    If Calcavecchia plays Friday morning , he knows the importance of starting well 

    for verb plays, return False:
        subjects is Calcavecchia
        but Friday morning is a npadvmod(as adverbial modifier) not object
    '''
    if is_passive_verb(v):
        rights = [x for x in v.rights if
                  (x.dep_ in OBJECT_DEPS or x.dep_ in {"prep", "agent"})
                  and x.dep_ not in {"npadvmod"}]
    else:
        rights = [
            x for x in v.rights if
            x.dep_ in OBJECT_DEPS
            and x.dep_ not in {"npadvmod"}]

    # Direct object
    if noun in rights:
        print("Find direct object in right children")
        return True

    # Continuous verb
    if obj_in_continuos_verb(noun, v):
        print("Find object in continous verb")
        return True

    # Preposition object (often occurs in passive sentence)
    if obj_in_preposition_check(noun, rights):
        print("Find objects in prepositions and its conjunction.")
        return True

    print("obj not found")
    return False


def RulesCheck(S: str, V: str, O: str, doc: Doc) -> bool:
    '''
    Return is valide SVO triplet or not

    Algorithm
    ---------
    Check subject is in the left side of verb (recursively)
    Check object is in the right side of verb (recursively)
        - if the verb is auxpass, find object in perposition
    '''
    valid, doc_S, doc_V, doc_O = ExtractDocIndexArr(doc, S, V, O)
    # Type checked failed
    if not valid:
        print("Not valid when type check")
        return False

    # Search in tree
    print("Parse", doc_S, doc_V, doc_O)
    for v in doc_V:
        sub_ok, obj_ok = False, False
        for s in doc_S:
            if SubjectCheck(s, v):
                sub_ok = True
                break
        for o in doc_O:
            if ObjectCheck(o, v):
                obj_ok = True
                break
        if sub_ok and obj_ok:
            return True
    return False


def main():
    path = "answer_trf_invert_auxfix.csv"
    labels = []
    df = read_CSV("data.csv")
    df = preprocessCSV(df)
    for i in range(len(df)):
        if i % 20 == 0:
            print("Perform on row {}".format(i))
        S, V, O, SENT = df.iloc[i, 1], df.iloc[i, 2],\
            df.iloc[i, 3], df.iloc[i, 4]
        valid = RulesCheck(S, V, O, NLP(SENT))
        if valid:
            labels.append(1)
        else:
            labels.append(0)
    print(Counter(labels))
    df["label"] = labels
    df.to_csv("debug/"+path, index=False)
    answer = df[["id", "label"]]
    answer.to_csv(path, index=False)


def debug():
    '''
    Debug sentences
    '''
    # doc = NLP("i asked the armenian native if the neighborhood kids or anybody else ever poked fun at the sign , which is basically a bart simpson dream come true .")
    # S, V, O = "i", "asked", "fun"
    # S, V, O = "anybody", "poked", "fun"
    # doc = NLP("A Spanish official , who had just finished a siesta and seemed not the least bit tense , offered what he believed to be a perfectly reasonable explanation for why the portable facilities were n't in service .")
    # S, V, O = "he","be","explanation"
    doc = NLP("she gave me a book, kissed me, and hugged me.")
    S, V, O = "she","gave","book"
    show_tree(doc)
    print(RulesCheck(S, V, O, doc))


if __name__ == "__main__":
    # main()
    debug()
