# Author: Yu-Lun Hsu
# Student ID: 1234567
# HW ID: hw2
# Due Date: 01/30/2022

from typing import List, Tuple, Union
import pandas as pd
import spacy
from spacy import tokens

NLP = spacy.load("en_core_web_sm")

# https://blog.csdn.net/u010087338/article/details/121055591
OBJECT_DEPS = {"dobj", "attr", "dative", "oprd"}
SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "agent", "expl"}
CONJ_DEP = {"CCONJ", "CONJ"}  # and or but
VERB_POS = {"AUX", "VERB"}  # is has will should


def readCSV(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


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
    

def findSubjectfromVerb(tokens: List[tokens.Token]):
    subs = []
    for token in tokens:
        if token.dep_ in SUBJECT_DEPS:
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
        if len(subs) == 0 and len(objs) == 0:
            continue
        svos.append((subs, v.text, objs))
    return svos


def FindSVO(doc: tokens.doc.Doc) -> List[Tuple[str, str, str]]:
    result = []
    sents = SplitBySent(doc)
    for sent in sents:
        sentence = SplitByCCONJ(sent)
        for s in sentence:
            svos = FindVerbs(s)
            result.extend(svos)
    return result


def Judge(pred:Tuple[str,str,str], actuals:List[Tuple[str,str,str]])->int:
    for actual in actuals:
        same = True
        for i in range(3):
            # if len(actual[i])>0:
            print("Compare => pred:{:25s} actual:{:25s}".format(pred[i],actual[i]))
            if actual[i] not in pred[i]:
                same = False
        print("Result: {}".format(same))
        print()
        if same: return 1
    return 0


def Parsing(df: pd.DataFrame, output_path: str, dry_run: bool=True):
    assert output_path.find(".csv")!=-1, "output path should be csv file"
    column_id = []
    column_label = []
    for i in range(len(df)):
        id = int(df.iloc[i, 0])
        pred = df.iloc[i,[1,2,3,]]
        sent = NLP(str(df.iloc[i,4]))
        actuals = FindSVO(sent)
        # Judge
        if(dry_run): print("=================================START===============================")
        if(dry_run): print(sent)
        ok = Judge(pred, actuals)
        column_id.append(id)
        column_label.append(ok)
        if(dry_run): print("=================================END({})===============================".format(ok),end="\n\n")
    if not dry_run:
        # export to csv
        pd.DataFrame({
            "id": column_id,
            "label": column_label,
        }).to_csv(output_path, index=False)
    else: 
        # export to debug
        df["same"] = column_label
        df = df[['id','same','S','V','O','sentence']]
        df.to_csv('debug/'+output_path, index=False)




if __name__ == "__main__":
    df = readCSV("data.csv")
    Parsing(df, "output_test_ignore.csv")