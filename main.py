from itertools import product
import json
import os
from typing import Dict, List

from tqdm import tqdm

from lib.vectorSpace import VectorSpace


vecSpace = VectorSpace()


print("adding EnglishNews to corpus...")
vecSpace.addDocsInFolder(
    os.path.join(os.path.dirname(__file__), "EnglishNews"), lang="en"
)
print("Corpus created!\n")

query = (
    input("Type query for problem 1 and 2 here (blank for 'Typhoon Taiwan war')\n> ")
    or "Typhoon Taiwan war"
)

print("####### problem 1 #######")

for similarity, weighting in product(["cos", "euclidean"], ["tf", "tf-idf"]):
    rankings = vecSpace.queryByText(query, weighting, similarity, "en", 10)
    print(weighting.upper(), "Cosine" if similarity == "cos" else "Euclidean")
    print("NewsID\t\tScore")
    for ranking in rankings:
        print(os.path.basename(ranking[0]), ranking[1], sep="\t")
    print("--------------------------------------")

print("####### problem 2 #######")

for similarity, weighting in product(["cos", "euclidean"], ["tf", "tf-idf"]):
    queryVec = vecSpace.buildTfVecFromText(query, "en")
    ranking = vecSpace.queryByVector(queryVec, weighting, similarity, 1)[0]
    with open(os.path.join(os.path.dirname(__file__), "EnglishNews", ranking[0])) as f:
        feedback = vecSpace.preProc.tokenize((f.read(),), "en")
    feedback_pos = filter(
        lambda x: x[1].startswith(("N", "V")),
        next(vecSpace.preProc.pos(feedback, "en")),
    )
    feedback = " ".join(map(lambda x: x[0], feedback_pos))
    queryVec = queryVec + vecSpace.buildTfVecFromText(feedback, "en") * 0.5

    rankings = vecSpace.queryByVector(queryVec, weighting, similarity, 10)
    print(weighting.upper(), "Cosine" if similarity == "cos" else "Euclidean")
    print("NewsID\t\tScore")
    for ranking in rankings:
        print(os.path.basename(ranking[0]), ranking[1], sep="\t")
    print("--------------------------------------")

print("####### problem 3 #######")

print("creating new corpus and adding ChineseNews to corpus...")
vecSpace = VectorSpace()
vecSpace.addDocsInFolder(
    os.path.join(os.path.dirname(__file__), "ChineseNews"), lang="zh"
)
print("Corpus created!\n")

query = (
    input("Type query for problem 3 here (blank for '資安 遊戲')\n> ") or "資安 遊戲"
)

for similarity, weighting in product(["cos"], ["tf", "tf-idf"]):
    rankings = vecSpace.queryByText(query, weighting, similarity, "zh", 10)
    print(weighting.upper(), "Cosine" if similarity == "cos" else "Euclidean")
    print("NewsID\t\tScore")
    for ranking in rankings:
        print(os.path.basename(ranking[0]), ranking[1], sep="\t")
    print("--------------------------------------")

print("####### problem 4 #######")

print("creating new corpus and adding smaller_dataset to corpus...")
vecSpace = VectorSpace()
vecSpace.addDocsInFolder(
    os.path.join(os.path.dirname(__file__), "smaller_dataset", "collections"), lang="en"
)
print("Corpus created!")
print("Reading answers from smaller_dataset/rel.tsv...")
with open(os.path.join(os.path.dirname(__file__), "smaller_dataset", "rel.tsv")) as f:
    answers: Dict[str, List[int]] = {
        line[0]: json.loads(line[1]) for line in map(lambda x: x.strip().split("\t"), f)
    }

nQueries = len(
    os.listdir(os.path.join(os.path.dirname(__file__), "smaller_dataset", "queries"))
)
print("Start calculating similarity scores")
for similarity, weighting in product(["cos", "euclidean"], ["tf", "tf-idf"]):
    MRR = 0
    MAP = 0
    recall = 0
    print(weighting.upper(), "Cosine" if similarity == "cos" else "Euclidean")
    for queryFName in tqdm(
        os.listdir(
            os.path.join(os.path.dirname(__file__), "smaller_dataset", "queries")
        )
    ):
        if queryFName.endswith(".txt"):
            with open(
                os.path.join(
                    os.path.dirname(__file__), "smaller_dataset", "queries", queryFName
                )
            ) as f:
                query = f.read()
            rankings = vecSpace.queryByText(query, weighting, similarity, "en", 10)
            corrects = 0
            precision = 0
            mrrFound = False
            for i, ranking in enumerate(rankings):
                if (
                    int(os.path.basename(ranking[0]).split(".")[0][1:])
                    in answers[queryFName.split(".")[0]]
                ):
                    # the retrieved document is relevant
                    if not mrrFound:
                        MRR += 1.0 / (i + 1)
                        mrrFound = True
                    corrects += 1
                    precision += corrects / (i + 1)
            if corrects > 0:
                MAP += precision / corrects
            recall += corrects / len(answers[queryFName.split(".")[0]])
    MRR /= nQueries
    MAP /= nQueries
    recall /= nQueries

    print("MRR@10\t", MRR, sep="\t")
    print("MAP@10\t", MAP, sep="\t")
    print("RECALL@10", recall, sep="\t")
    print("--------------------------------------")
