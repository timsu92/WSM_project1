from collections import defaultdict
import heapq
import itertools
from typing import ClassVar, DefaultDict, Iterable, List, Literal, Tuple

import numpy as np
from numpy.typing import NDArray

from lib.preprocess import Preprocessor
from utils.BiDirectionalDict import Bidict
from utils.io import folderTxtIterator
from utils.math import cosine


class VectorSpace:
    preProc: ClassVar = Preprocessor()

    def __init__(self) -> None:
        self.vectorKeywordIndex = Bidict[str, int]()

        self.tf: DefaultDict[str, DefaultDict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        """self.tf[document][word] => count"""

        self.idf: DefaultDict[int, int] = defaultdict(int)
        """self.idf[word] => count of documents"""

    def addDocs(
        self, docs: Iterable[Tuple[str, str]], lang: Literal["en", "zh", "both"]
    ):
        """@param docs: docName => docText"""
        docs, docs2 = itertools.tee(docs)  # copy the iterator
        for docName, docTokens in zip(
            (docName for docName, _ in docs),
            self.preProc.tokenize((docText for _, docText in docs2), lang),
        ):
            assert docName not in self.tf.keys()
            addedWords = set()
            for token in docTokens:
                if token not in self.vectorKeywordIndex:
                    self.vectorKeywordIndex[token] = len(self.vectorKeywordIndex)
                token = self.vectorKeywordIndex[token]
                if not token in addedWords:
                    addedWords.add(token)
                    self.idf[token] += 1
                self.tf[docName][token] += 1

    def addDocsInFolder(self, folderPath: str, lang: Literal["en", "zh", "both"]):
        self.addDocs(folderTxtIterator(folderPath), lang)

    def text2vec(self, texts: List[str], lang: Literal["en", "zh", "both"]):
        for text in self.preProc.tokenize(texts, lang):
            tokens: List[int] = []
            for token in text:
                if token not in self.vectorKeywordIndex:
                    self.vectorKeywordIndex[token] = len(self.vectorKeywordIndex)
                tokens.append(self.vectorKeywordIndex[token])
            yield tokens

    def buildVecFromCount(self, docName: str):
        if docName not in self.tf.keys():
            raise ValueError(f"docName {docName} not in tf")
        vec = np.zeros(len(self.vectorKeywordIndex), dtype=np.uint16)
        for word, count in self.tf[docName].items():
            vec[word] = count
        return vec

    def buildTfVecFromText(self, text: str, lang: Literal["en", "zh", "both"]):
        vec = np.zeros(len(self.vectorKeywordIndex), dtype=np.uint16)
        for docTokens in self.preProc.tokenize([text], lang):
            for word in docTokens:
                try:
                    vec[self.vectorKeywordIndex[word]] += 1
                except KeyError:
                    pass
        return vec

    def similarity(
        self,
        queryVec: NDArray[np.uint],
        docName: str,
        weighting: Literal["tf", "tf-idf"],
        comparer: Literal["cos", "euclidean"],
    ):
        docVec = self.buildVecFromCount(docName)
        if weighting == "tf-idf":
            queryVec = queryVec * np.log2(
                np.full(len(self.idf), len(self.tf.keys())) / list(self.idf.values())
            )
            docVec = docVec * np.log2(
                np.full(len(self.idf), len(self.tf.keys())) / list(self.idf.values())
            )
            # queryVec = queryVec.copy()
            # for i in range(len(queryVec)):
            #     queryVec[i] *= np.log2(len(self.tf.keys()) / self.idf[i])
            #     docVec[i] *= np.log2(len(self.tf.keys()) / self.idf[i])

        if comparer == "cos":
            return cosine(queryVec, docVec)
        elif comparer == "euclidean":
            return np.linalg.norm(queryVec - docVec)
        else:
            raise ValueError(f"comparer {comparer} not supported")

    def queryByVector(
        self,
        query: NDArray,
        weighting: Literal["tf", "tf-idf"],
        comparer: Literal["cos", "euclidean"],
        topK: int,
    ):
        if weighting == "tf-idf":
            for i in range(len(query)):
                query[i] *= np.log2(len(self.tf.keys()) / self.idf[i])
        if comparer == "cos":
            return heapq.nlargest(
                topK,
                (
                    (docName, self.similarity(query, docName, weighting, comparer))
                    for docName, docVec in self.tf.items()
                ),
                lambda x: x[1],
            )
        else:
            return heapq.nsmallest(
                topK,
                (
                    (docName, self.similarity(query, docName, weighting, comparer))
                    for docName, docVec in self.tf.items()
                ),
                lambda x: x[1],
            )

    def queryByText(
        self,
        query: str,
        weighting: Literal["tf", "tf-idf"],
        comparer: Literal["cos", "euclidean"],
        lang: Literal["en", "zh", "both"],
        topK: int,
    ):
        queryVec = self.buildTfVecFromText(query, lang)
        return self.queryByVector(queryVec, weighting, comparer, topK)
