# Author: Yu-Lun Hsu
# Student ID: 0716235
# HW ID: hw2
# Due Date: 01/30/2022

'''
Paper Reference: 
https://www.researchgate.net/publication/228905420_Triplet_extraction_from_sentences
'''

from typing import List, Literal, Set, Tuple, Union
import pandas as pd
import spacy
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span
from spacy.tokens import Token

DEBUGMODE = False

SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"}
OBJECTS_DEPS = {"dobj", "dative", "attr", "oprd", "pobj"}

NLP = spacy.load("en_core_web_trf")


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
    沒有主詞的句子有可能是因為他是補充說明，主詞被藏在祖先節點的左子樹之中
    '''
    current = verb.head
    while current.pos_ != "VERB" and current.pos_ not in {"NOUN", "PRON", "PROPN"} and current.head != current:
        current = current.head
    if current.pos_ == "VERB":
        subs = [t for t in current.lefts if t.dep_ == "SUB"]
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
    將抽象代名詞本身指向的事物回傳
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

    # 把WH之類的東西換成原本的
    for sub in subs:
        if sub.tag_ == "WP" and sub.pos_ == "PRON":
            implicit_subjects = find_implicit_subjects(sub)
            subs.append(implicit_subjects)
    return subs


def FindObjectsFromPrepositions(tokens: List[Token]):
    '''
    從介係詞中抽取obj
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
    從開放式從句補語尋找verb與對應的objs(這種句子的動詞沒有主語)

    Example
    ---------
    She looks very beautiful. (beautiful 用來修飾 looks 而不是 She)
    Sue asked George to respond to her offer. (respond 用來修飾 asked)
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
    解析名詞子句, 回傳
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
        5. find all objs (要可以替換代名詞, which who ...)
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
            subjects = FindSubjects(v)
            objects = FindObjects(v)
            log("Subject Find", subjects)
            log("Object Find", objects)
            for s in subjects:
                for o in objects:
                    log("Extract from", s, o, color="\033[92m")
                    subs = Expand(s, sent, visited)
                    objs = Expand(o, sent, visited)
                    svo.append(
                        (JoinTokens(subs), v.text, JoinTokens(objs)))
        svos.append(svo)
    return svos


def verb_validaty_check(V_token: Doc) -> bool:
    '''
    Check the given verb contains noun phrase
    '''
    for token in V_token:
        if token.pos_ in SUBJECT_DEPS or token.pos_ in OBJECTS_DEPS:
            return False
    return True


def verb_subset_check(V: str, solution_V: str) -> bool:
    '''
    Check solution_S is a subset of S
    '''
    given_set = set(V.split())
    expected_set = set(solution_V.split())
    return expected_set <= given_set


def subobj_check(get: str, solution: str) -> bool:
    '''
    Check get is a subset of solution
    '''
    if get not in solution:
        return False
    return True


def noun_check(doc: Doc) -> bool:
    '''
    Check if noun, pron propn in given doc
    '''
    for token in doc:
        if token.pos_ in {"NOUN", "PRON", "PROPN"}:
            return True
    return False


def CompareSimilarity(S: str, V: str, O: str, answerList: List[List[str]], threshold: float = 0.9) -> Literal[0, 1]:
    '''
    Compare similarity
    '''
    S_doc = NLP(S)
    V_doc = NLP(V)
    O_doc = NLP(O)

    # Check Verb is valid
    if not verb_validaty_check(V_doc):
        return 0

    if not noun_check(S_doc):
        return 0

    if not noun_check(O_doc):
        return 0

    for sentence in answerList:
        for solution in sentence:
            S_check, V_check, O_check = False, False, False
            S_value = S_doc.similarity(NLP(solution[0]))
            V_value = V_doc.similarity(NLP(solution[1]))
            O_value = O_doc.similarity(NLP(solution[2]))
            # Check Subject
            if S_value > threshold:
                S_check = True
            elif subobj_check(S, solution[0]):
                S_check = True
            # Check Verb
            if V_value > threshold:
                V_check = True
            elif verb_subset_check(V, solution[1]):
                V_check = True
            # Check Object
            if O_value > threshold:
                O_check = True
            elif subobj_check(O, solution[2]):
                O_check = True
            # If three check is passed, return 1
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
    path = "answer_enhance_trf_threshold_90_check.csv"
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
    if DEBUGMODE:
        df.to_csv("debug/"+path, index=False)
    else:
        answer = df[["id", "label"]]
        answer.to_csv(path, index=False)


if __name__ == "__main__":
    main()
