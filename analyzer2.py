import sys

import pandas as pd

import configuration
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np


def plot_word_counts(output_folder, foldername, words_freq, k):
    plt.clf()
    fig = plt.figure(figsize=(16, 9))
    plt.barh(-np.arange(k), words_freq[:k, 1].astype(float), height=0.8)
    plt.yticks(ticks=-np.arange(k), labels=words_freq[:k, 0])
    plt.title(foldername)
    path = os.path.join(output_folder, foldername + '/wordcount.png')
    plt.savefig(path)
    plt.close(fig)


def main():
    arg_length = len(sys.argv)
    if arg_length < 2:
        raise Exception("Please provide the properties file as an argument")

    configuration.load_properties(sys.argv[1])

    split = True if configuration.config["split.split"].data == "true" else False

    import repository
    output_folder = "analysis2"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print("Loading repositories... (this can take a while)")
    repositories = repository.Repositories()
    data = {}
    target = []

    for repo in repositories.data.values():
        all_data = ""
        all_split_data = ""
        for analysis_type in repo.data.keys():
            data.setdefault(analysis_type, []).append(repo.data[analysis_type])

            if analysis_type.endswith(configuration.config["split.suffix"].data):
                all_split_data = " ".join([all_split_data, repo.data[analysis_type]])
            else:
                all_data = " ".join([all_data, repo.data[analysis_type]])

        data.setdefault("all", []).append(all_data)
        data.setdefault("all_" + configuration.config["split.suffix"].data, []).append(all_split_data)
        target.append(repo.target)

    number_of_topics = len(set(target))
    print("Loaded " + str(len(repositories.data)) + " repositories with " + str(number_of_topics) + " total topics!")
    for analysis_type in data.keys():
        print("Analyzing " + analysis_type)
        # basic data analysis

        folder_name = str(analysis_type)

        if not os.path.exists(os.path.join(output_folder, folder_name)):
            os.makedirs(os.path.join(output_folder, folder_name))

        vectorizer = TfidfVectorizer(min_df=2, max_df=0.50)
        bow = vectorizer.fit_transform(data[analysis_type])

        # word importance plotting
        words_freq = list(zip(vectorizer.get_feature_names_out(), [x[0] for x in bow.sum(axis=0).T.tolist()]))
        words_freq = np.array(sorted(words_freq, key=lambda x: x[1], reverse=True))

        plot_word_counts(output_folder, folder_name, words_freq, 20)

        # LSA
        lsa = TruncatedSVD(n_components=number_of_topics)
        lsa_matrix = lsa.fit_transform(bow)
        lsa_target = np.argmax(lsa_matrix, axis=1).tolist()
        lsa_target_percentages = np.amax(lsa_matrix, axis=1).tolist()

        # TSNE perplexity 20 orig
        tsne = TSNE(n_components=2, perplexity=20, learning_rate=100, n_iter=2000, random_state=0, angle=0.75)
        tsne_mat = tsne.fit_transform(lsa_matrix)

        # Plotting setup
        fig = plt.figure(layout="constrained")
        fig.set_size_inches((18.5, 10.5), forward=False)
        gs = GridSpec(1, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])

        # Plot axes 1
        sns.scatterplot(x=tsne_mat[:, 0], y=tsne_mat[:, 1], hue=target, ax=ax1, legend=False, palette="hls")
        ax1.set_title("t-SNE with manual labels")

        # Plot axes 2
        sns.scatterplot(x=tsne_mat[:, 0], y=tsne_mat[:, 1], hue=lsa_target, ax=ax2, legend=False, palette="hls", size=lsa_target_percentages, sizes=(10, 100))
        ax2.set_title("t-SNE with LSA Labels")

        # Export
        plt.suptitle(folder_name)
        path = os.path.join(output_folder, folder_name + '/t-SNE Plot.png')
        plt.savefig(path, dpi=200)
        plt.close(fig)

        df = pd.DataFrame(lsa_matrix, columns=["Topic " + str(x) for x in range(number_of_topics)])
        df.insert(0, "repository", repositories.data.keys())
        df.insert(1, "target", target)
        df.insert(2, "lsa", lsa_target)
        path = os.path.join(output_folder, folder_name + '/LSA Output.csv')
        df.to_csv(path)


if __name__ == "__main__":
    main()
