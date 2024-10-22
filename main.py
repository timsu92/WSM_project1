from itertools import product
from multiprocessing.spawn import prepare
import os
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
    feedback_pos = filter(lambda x: x[1].startswith(("N", "V")), next(vecSpace.preProc.pos(feedback, "en")))
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
    input("Type query for problem 3 here (blank for '資安 遊戲')\n> ")
    or "資安 遊戲"
)

for similarity, weighting in product(["cos"], ["tf", "tf-idf"]):
    rankings = vecSpace.queryByText(query, weighting, similarity, "zh", 10)
    print(weighting.upper(), "Cosine" if similarity == "cos" else "Euclidean")
    print("NewsID\t\tScore")
    for ranking in rankings:
        print(os.path.basename(ranking[0]), ranking[1], sep="\t")
    print("--------------------------------------")