# Author: Yu-Lun Hsu
# Student ID: 0716235
# HW ID: hw2
# Due Date: 01/30/2022

from typing import List, Literal, Set, Tuple, Union
import pandas as pd
import spacy
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span
from spacy.tokens import Token

DEBUGMODE = True

SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"}
OBJECTS_DEPS = {"dobj", "dative", "attr", "oprd", "pobj"}

# https://www.researchgate.net/publication/228905420_Triplet_extraction_from_sentences
NLP = spacy.load("en_core_web_sm")


class Color:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'


def log(*args, **kwargs):
    '''
    print colorize debug log.

    colorset:
        - yellow: yellow, warning (default)
        - red: red, error, danger
        - blue: info, blue
        - green:  success, ok, green
    Example
    ---------
    >>> log("a","b","c", color='danger')
    '''
    if DEBUGMODE:
        color = ""
        tag = ""
        if "color" in kwargs and isinstance(kwargs["color"], str):
            kc = kwargs["color"].lower()
            if kc in ["yellow", "warning"]:
                color = Color.YELLOW
                tag = "WARN"
            elif kc in ["red", "error", "danger"]:
                color = Color.RED
                tag = "ERROR"
            elif kc in ["blue", "info"]:
                color = Color.BLUE
                tag = "INFO"
            elif kc in ["success", "ok", "green"]:
                color = Color.GREEN
                tag = "PASS"
        else:
            tag = "WARN"
            color = Color.YELLOW
        args_str = ' '.join(map(str, args))
        print("{}[{}] {}{}".format(color, tag, args_str, Color.RESET))


def show_tree(doc: Doc):
    '''
    Show word ,tag, dep, pos, head, left, right of given doc
    '''
    print('%10s %5s %10s %10s %10s %10s %10s' %
          ("word", "tag", "dep", "pos", "head", "left", "right"))
    print('=================================================================================================')
    for word in doc:
        print('%10s %5s %10s %10s %10s %10s %10s' %
              (word.text, word.tag_, word.dep_, word.pos_, word.head.text, list(word.lefts), list(word.rights)))


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


def FindVerb(token: Token) -> List[Token]:
    '''
    Find verb in sentence, first find a non aux verb. If the set is empty,
    find the aux verb instead.
    '''
    verbs = [t for t in token if is_verb(t)]
    return verbs


def FindSubjectsInConj(subs: List[Token]):
    '''
    Find subjects which connects by conj recursively.
    Example
    --------
    Alan and Bob and Mike likes that girl.
    verbs = ["likes", ]
    subjects = ["Alan" ]
    we expected to get subjects = ["Alan","Bob","Mike"]
    '''
    extra = []
    for s in subs:
        right_children = list(s.rights)
        deps = {t.dep_ for t in right_children}
        if "cc" in deps:
            # recursively find subject from right chlidren
            subjects = [
                t for t in right_children if t.dep_ in SUBJECT_DEPS or t.dep_ == "conj"]
            subjects.extend(FindSubjectsInConj(subjects))
            if len(subjects) > 0:
                extra.extend(subjects)
    return extra


def find_hidding_subjects(verb: Token) -> List[Token]:
    '''
    Explain
    -----------
    If the sentence is: 

    `"A Spanish official , who had just finished a siesta and seemed not the least bit tense,\
    offered what he believed to be a perfectly reasonable explanation for why the portable facilities were n't in service ."`

    expected subjects for verb "seemed" are ["official"]

    but "seemed" only has right children "tense"(object).

    the subject of "seemed" is "official", which is the verb's grandparent.

    So, we must find the source by tracing ancestors.

    Algorithm 
    -----------
    ```
    def find_hidding_subjects(verb)->List[Token]:
        current = verb.head
        if IS_NOT_VERB(current) and IS_NOT_NOUNS(current) and HAVE_HEAD(current):
            current = current.head
        if IS_VERB(current):
            subs = [all subjects in left children]
            if NOT_EMPTY(subs):
                more = FindSubjectsInConj(subs)
                subs.extend(more)
                return subs
            elif HAVE_HEAD(current)
                return  find_hidding_subjects(current)
        elif IS_NOUN_OR_PRON(current):
            return current as a list
        else:
            return []
    ```
    '''
    current = verb.head
    while current.pos_ != "VERB" and current.pos_ not in {"NOUN", "PRON", "PROPN"} and current.head != current:
        current = current.head
    if current.pos_ == "VERB":
        subs = [t for t in current.lefts if t.dep_ in SUBJECT_DEPS]
        if len(subs) > 0:
            subs.extend(FindSubjectsInConj(subs))
            return subs
        elif current.head != current:
            return find_hidding_subjects(current)
    elif current.pos_ in {"NOUN", "PRON"}:
        return [current]
    return []


def find_implicit_subjects(sub: Token) -> Token:
    '''
    ?????????????????????????????????????????????
    '''
    current = sub
    while current.pos_ != "NOUN" and current.head != current:
        current = current.head
    return current


def FindSubjects(verb: Token):
    '''
    Algorithm
    -----------------
    find subjects in a verb's left children.
    if found subjects:
        find conjunction subjects recursively
    else:
        find  hidding objects since the head is in other sentences
    add the real NOUN clause for tag "WP"
    '''
    subs = [child for child in verb.lefts if child.dep_ in SUBJECT_DEPS]
    if len(subs) > 0:
        subs.extend(FindSubjectsInConj(subs))
    else:
        # find hiding subjects
        hidding_subs = find_hidding_subjects(verb)
        subs.extend(hidding_subs)

    # ???WH??????????????????????????????
    for sub in subs:
        if sub.tag_ == "WP" and sub.pos_ == "PRON":
            implicit_subjects = find_implicit_subjects(sub)
            subs.append(implicit_subjects)
    return subs


def FindObjectsFromPrepositions(tokens: List[Token]):
    '''
    ?????????????????????obj
    '''
    extra = []
    for t in tokens:
        if t.pos_ == "ADP" and t.dep_ in {"prep", "agent"}:
            objs = [r for r in t.rights if r.dep_ in OBJECTS_DEPS or (
                r.pos_ == "PRON")]
            extra.extend(objs)
    return extra


def FindObjectsFromXcomp(tokens: List[Token]) -> Union[Tuple[Token, List[Token]], Tuple[None, None]]:
    '''
    ??????????????????????????????verb????????????objs(?????????????????????????????????)

    Example
    ---------
    She looks very beautiful. (beautiful ???????????? looks ????????? She)
    Sue asked George to respond to her offer. (respond ???????????? asked)
    '''
    for t in tokens:
        if t.pos_ == "VERB" and t.dep_ == "xcomp":
            rights = list(t.rights)
            objs = [child for child in rights if child.dep_ in OBJECTS_DEPS]
            objs.extend(FindObjectsFromPrepositions(rights))
            if len(objs) > 0:
                return t, objs
    return None, None


def FindObjectsInConj(objs: List[Token]) -> List[Token]:
    '''
    Find objects which connects by conj recursively.

    Example
    ---------
    I love Namin and her sister.
    verbs = ["love"],
    objs = ["Namin],
    we expected to get ["Namin", "sister].
    '''
    extra = []
    for o in objs:
        right_children = list(o.rights)
        deps = {t.dep_ for t in right_children}
        if "cc" in deps:
            # recursively find subject from right chlidren
            objects = [
                t for t in right_children if t.dep_ in OBJECTS_DEPS or t.dep_ == "conj"]
            objects.extend(FindObjectsFromPrepositions(objects))
            if len(objects) > 0:
                extra.extend(objects)
    return extra


def FindObjectsNounClause(verb: Token) -> Union[Token, None]:
    ''' 
    ??????????????????, ??????
    Example
    --------
    The students thought that they had met us.
    INCluase = [that]
    '''
    for child in verb.rights:
        if child.dep_ == "ccomp":
            for left in child.lefts:
                if left.dep_ == "mark":
                    return left
    return None


def FindObjects(verb: Token) -> List[Token]:
    '''
    Find objects from the right children of verb
    '''
    right_children = list(verb.rights)

    objs = [t for t in right_children if t.dep_ in OBJECTS_DEPS]
    prep_obj = FindObjectsFromPrepositions(objs)
    objs.extend(prep_obj)

    noun_clause = FindObjectsNounClause(verb)
    if noun_clause != None:
        objs.append(noun_clause)

    new_verb, new_objs = FindObjectsFromXcomp(right_children)
    if new_verb != None and new_objs != None:
        if len(new_objs) > 0:
            objs.extend(new_objs)

    if len(objs) > 0:
        conj_obj = FindObjectsInConj(objs)
        objs.extend(conj_obj)

    return objs


def Expand(item: Token, sentence: Span, visited: Set) -> List[Token]:
    '''
    expand an object/subject in doc tree.
    A complete dispict of the subject or object must read in the order of (left, root, right).
    So I implement an inorder traversal of the tokenized tree.
    '''

    parts = []

    for l in item.lefts:
        if l.pos_ in {"CCONJ", "VERB"}:
            break
        element = Expand(l, sentence, visited)
        parts.extend(element)

    parts.append(item)

    for r in item.rights:
        if r.pos_ in {"CCONJ", "VERB"}:
            break
        parts.append(r)

    if len(parts) > 0:
        if hasattr(parts[-1], 'rights'):
            for x in parts[-1].rights:
                if x.pos_ == "DET" or x.pos_ in {"NOUN", "PRON"}:
                    if x.i not in visited:
                        visited.add(x.i)
                        parts.extend(Expand(x, sentence, visited))
                break
    return parts


def JoinTokens(tokens: List[Token]) -> str:
    return ' '.join([t.text for t in tokens])


def SVOParse(doc: Doc):
    '''
    Find and return SVO triplets. 

    Algorithm
    ----------
    1. split the string into sentences
    2. find verbs in a sentence
    3. according to the verb, find joined subjects from its lefthand
    if len(subjects) > 0:
        4. join the right of verb if it is conjunction verb (ex: she hug and kiss me.)
        5. find all objs (????????????????????????, which who ...)
    6. retreive detail dispict of subject and object
    7. append to svo list
    '''
    svos = []
    for sent in doc.sents:
        svo = []
        verbs = FindVerb(sent)
        log("Get Verbs: ", verbs)
        visited = set()
        for v in verbs:
            log("verb", v, color="info")
            subjects = FindSubjects(v)
            objects = FindObjects(v)
            log("Subject Find", subjects)
            log("Object Find", objects)
            for s in subjects:
                for o in objects:
                    subs = Expand(s, sent, visited)
                    objs = Expand(o, sent, visited)
                    svo.append(
                        (JoinTokens(subs), v.text, JoinTokens(objs)))
        svos.append(svo)
    return svos


def CompareSimilarity(S: str, V: str, O: str, answerList: List[List[str]], threshold: float = 0.8) -> Literal[0, 1]:
    '''
    Compare similarity between
    '''
    for sentence in answerList:
        S_check, V_check, O_check = False, False, False
        for solution in sentence:
            S_value = NLP(S).similarity(NLP(solution[0]))
            V_value = NLP(V).similarity(NLP(solution[1]))
            O_value = NLP(O).similarity(NLP(solution[2]))
            if S_value > threshold:
                S_check = True
            if V_value > threshold:
                V_check = True
            if O_value > threshold:
                O_check = True
        if S_check and V_check and O_check:
            return 1
    return 0


def read_CSV(path: str) -> pd.DataFrame:
    '''
    read csv file from given path
    '''
    assert path.find(".csv") != -1, "expected to be csv file"
    df = pd.read_csv(path)
    return df


def main():
    labels = []
    df = read_CSV("data.csv")
    for i in range(len(df)):
        if i % 20 == 0:
            print("Perform on row {}".format(i))
        S, V, O, sent = df.iloc[i, 1], df.iloc[i,
                                               2], df.iloc[i, 3], df.iloc[i, 4]
        answerList = SVOParse(NLP(str(sent)))
        label = CompareSimilarity(str(S), str(V), str(O), answerList)
        labels.append(label)
    df["label"] = labels
    answer = df[["id", "label"]]
    answer.to_csv("answer_enhance_threshold.csv", index=False)


if __name__ == "__main__":
    doc = NLP("A Spanish official , who had just finished a siesta and seemed not the least bit tense , offered what he believed to be a perfectly reasonable explanation for why the portable facilities were n't in service .")
    print('%10s %5s %10s %10s %10s %10s %10s' %
          ("word", "tag", "dep", "pos", "head", "left", "right"))
    print('=================================================================================================')
    for word in doc:
        print('%10s %5s %10s %10s %10s %10s %10s' %
              (word.text, word.tag_, word.dep_, word.pos_, word.head.text, list(word.lefts), list(word.rights)))
    answerList = SVOParse(doc)
    log(answerList, color="ok")
