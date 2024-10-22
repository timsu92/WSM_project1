import os
from typing import Iterable, List, Literal

import ckiptagger
import nltk

from lib.porterStemmer import PorterStemmer


class Preprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        with open(os.path.join(os.path.dirname(__file__), "eng.stop"), "r") as f:
            self.stopWords = set(f.read().split())
        if not os.path.isdir("./data"):
            print("downloading data for ckiptagger...")
            print("Will download and extract to 2.5GB in total...")
            ckiptagger.data_utils.download_data_url("./")
            os.remove("./data.zip")
        self.chTokenizer = ckiptagger.WS("./data")
        nltk.download('averaged_perceptron_tagger_eng')
        self.chPOS = ckiptagger.POS("./data")

    @staticmethod
    def clean(text: str):
        """remove any nasty grammar tokens from string"""
        return (
            text.replace(r"\s+", " ")
            .strip()
            .replace(",", "")
            .replace(".", "")
            .replace("，", " ")
            .replace("。", " ")
            .lower()
        )

    def removeStopWords(self, tokens: Iterable[str]):
        return (token for token in tokens if token not in self.stopWords)

    def tokenize(self, docs: Iterable[str], lang: Literal["en", "zh", "both"]):
        """tokenize text into words and stem them
        `both` lang is considered `zh` mode"""
        docs = (self.clean(t) for t in docs)
        if lang == "en":
            docTokens = (t.split(" ") for t in docs)
            docTokens = (self.removeStopWords(doc) for doc in docTokens)
            for doc in docTokens:
                yield [self.stemmer.stem(word, 0, len(word) - 1) for word in doc]
        else:
            docTokens = self.chTokenizer(list(docs))
            for doc in docTokens:
                tokens = []
                for token in doc:
                    if token.isascii():
                        tokens.extend(
                            self.stemmer.stem(t, 0, len(t) - 1)
                            for t in token.split(" ")
                        )
                    else:
                        tokens.append(token)
                yield tokens

    def pos(self, docTokens: Iterable[List[str]], lang: Literal["en", "zh", "both"]):
        """@pre: tokenize"""
        if lang == "en":
            return (nltk.pos_tag(tok) for tok in docTokens)
        else:
            raise ValueError(f"lang {lang} not supported")
