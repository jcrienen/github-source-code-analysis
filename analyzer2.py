import sys
import configuration
import os

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import seaborn as sns
from umap import UMAP

import matplotlib.pyplot as plt
import numpy as np


def plot_word_counts(output_folder, filename, words_freq, k):
    plt.clf()
    plt.figure(figsize=(16, 9))
    plt.barh(-np.arange(k), words_freq[:k, 1].astype(float), height=0.8)
    plt.yticks(ticks=-np.arange(k), labels=words_freq[:k, 0])
    plt.title(filename)
    path = os.path.join(output_folder, filename + '_wordcount.png')
    plt.savefig(path)


def main():
    arg_length = len(sys.argv)
    if arg_length < 2:
        raise Exception("Please provide the properties file as an argument")

    configuration.load_properties(sys.argv[1])

    split = True if configuration.config["split.split"].data == "true" else False
    splitsuffix = ("_" + configuration.config["split.suffix"].data) if split else ""

    import repository
    output_folder = "analysis"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for analysis_type in repository.AnalysisType:
        # basic data analysis
        repositories = repository.Repositories(analysis_type, split)
        filename = str(analysis_type.value) + splitsuffix
        data = [repo.data for repo in repositories.data.values()]
        target = [repo.target for repo in repositories.data.values()]
        vectorizer = TfidfVectorizer(min_df=1, max_df=.50)
        bow = vectorizer.fit_transform(data)

        # word importance plotting
        words_freq = list(zip(vectorizer.get_feature_names_out(), [x[0] for x in bow.sum(axis=0).T.tolist()]))
        words_freq = np.array(sorted(words_freq, key=lambda x:x[1], reverse=True))

        plot_word_counts(output_folder, filename, words_freq, 20)

        # lsa
        lsa = TruncatedSVD(n_components=2)
        lsa_matrix = lsa.fit_transform(bow)

        # TSNE
        tsne = TSNE(n_components=2, perplexity=20)
        tsne_mat = tsne.fit_transform(lsa_matrix)

        plt.clf()
        plt.figure(figsize=(16, 9))
        sns.scatterplot(x=tsne_mat[:, 0], y=tsne_mat[:, 1], hue=target)
        path = os.path.join(output_folder, filename + '_tsne.png')
        plt.title(filename)
        plt.savefig(path)

        # UMAP
        embedding = UMAP(n_neighbors=150, min_dist=0.5, random_state=12).fit_transform(lsa_matrix)
        plt.clf()
        plt.figure(figsize=(16, 9))
        sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=target)
        path = os.path.join(output_folder, filename + '_umap.png')
        plt.title(filename)
        plt.savefig(path)


if __name__ == "__main__":
    main()
