from itertools import product
import os
from lib.vectorSpace import VectorSpace


vecSpace = VectorSpace()
query = "Typhoon Taiwan war"

####### question 1 #######

print("adding EnglishNews to corpus...")
vecSpace.addDocsInFolder(
    os.path.join(os.path.dirname(__file__), "EnglishNews"), lang="en"
)
print("Corpus created!\n")
for similarity, weighting in product(["cos", "euclidean"], ["tf", "tf-idf"]):
    rankings = vecSpace.query(query, weighting, similarity, "en", 10)
    print(weighting.upper(), "Cosine" if similarity == "cos" else "Euclidean")
    print("NewsID\t\tScore")
    for ranking in rankings:
        print(os.path.basename(ranking[0]), ranking[1], sep="\t")
    print("--------------------------------------")
