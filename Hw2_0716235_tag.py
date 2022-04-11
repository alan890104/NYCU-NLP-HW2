# Author: Yu-Lun Hsu
# Student ID: 1234567
# HW ID: hw2
# Due Date: 01/30/2022

from operator import index
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


def Judge(S:str, V:str, O:str, sent: str)->int:
    doc = NLP(sent)
    objExist = False
    subExist = False
    for token in doc:
        if token.dep_ in OBJECT_DEPS and token.head.text in V and token.text in O:
            objExist = True
        if token.dep_ in SUBJECT_DEPS and token.head.text in V and token.text in S:
            subExist = True
    if objExist and subExist:
        return 1
    else: 
        return 0


def Parsing(df: pd.DataFrame):
    labels = []
    for i in range(len(df)):
        S,V,O,sent = df.iloc[i,1],df.iloc[i,2],df.iloc[i,3],df.iloc[i,4]
        label = Judge(str(S),str(V),str(O),sent)
        labels.append(label)        
    df["label"] = labels
    answer = df[["id","label"]]
    answer.to_csv("answer.csv",index=False)



if __name__ == "__main__":
    df = readCSV("data.csv")
    Parsing(df)