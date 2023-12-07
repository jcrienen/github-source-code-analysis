import sys
import configuration
import os

from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import umap


def generate_bag_of_words(repositories):
    data = [repo.data for repo in repositories.data.values()]
    vectorizer = TfidfVectorizer(min_df=1)
    bow = vectorizer.fit_transform(data)
    print(bow.shape)

    return vectorizer, bow, data


def create_svd(bow):
    svd = TruncatedSVD(n_components=20, algorithm="randomized", n_iter=100, random_state=122)
    topics = svd.fit_transform(bow)
    print("Topics: " + str(len(svd.components_)))

    return svd, topics


def create_pca(bow):
    pca = PCA(n_components=2).fit_transform(bow.toarray())
    return pca


def plot(pca, title):
    df = pd.DataFrame(pca, columns=list('xy'))
    ax = df.plot(kind='scatter', x='x', y='y', xlim=(-1, 1), ylim=(-1, 1), title=title)


def plot2(data, topics):
    embedding = umap.UMAP(n_neighbors=150, min_dist=0.5, random_state=12).fit_transform(topics)

    plt.figure(figsize=(7, 5))
    plt.scatter(embedding[:, 0], embedding[:, 1],
                c=np.array(data),
                s=10,  # size
                edgecolor='none'
                )
    plt.show()


def main():
    arg_length = len(sys.argv)
    if arg_length < 2:
        raise Exception("Please provide the properties file as an argument")

    configuration.load_properties(sys.argv[1])

    split = True if configuration.config["split.split"].data == "true" else False
    splitsuffix = ("_" + configuration.config["split.suffix"].data) if split else ""

    import repository

    for analysis_type in repository.AnalysisType:
        print(analysis_type.name + ": ")
        repositories = repository.Repositories(analysis_type, split)
        vectorizer, bow, data = generate_bag_of_words(repositories)
        svd, topics = create_svd(bow)

        terms = vectorizer.get_feature_names_out()

        for i, comp in enumerate(svd.components_):
            terms_comp = zip(terms, comp)
            sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)[:7]
            print("Topic " + str(i) + ": ", end="")
            for t in sorted_terms:
                print(t[0], end="")
                print(" ", end="")
            print()

        plot2(data, topics)

        # pca = create_pca(bow)
        # plot(pca, analysis_type.value + splitsuffix)
        # output_folder = configuration.config["output-folder"].data
        # path = os.path.join(output_folder, str(analysis_type.value) + splitsuffix + '.png')
        # plt.savefig(path)


if __name__ == "__main__":
    main()
