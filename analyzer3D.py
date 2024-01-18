import sys

import pandas as pd
from matplotlib.colors import ListedColormap

import configuration
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import seaborn as sns

from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np


def main():
    arg_length = len(sys.argv)
    if arg_length < 2:
        raise Exception("Please provide the properties file as an argument")

    configuration.load_properties(sys.argv[1])

    split = True if configuration.config["split.split"].data == "true" else False

    import repository
    output_folder = "analysis_3d"
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

    repos = np.array(list(repositories.data.keys()))
    number_of_topics = len(set(target))
    print("Loaded " + str(len(repositories.data)) + " repositories with " + str(number_of_topics) + " total topics!")

    queries = []
    with open("queries.txt", "r") as f:
        for line in f:
            line = line.strip()
            queries.append(line)

    lsa_combined = {}
    for analysis_type in data.keys():
        # basic data analysis
        print("Analyzing " + analysis_type + ", number of topics: " + str(number_of_topics))
        folder_name = str(analysis_type)

        if not os.path.exists(os.path.join(output_folder, folder_name)):
            os.makedirs(os.path.join(output_folder, folder_name))

        vectorizer = TfidfVectorizer(min_df=2, max_df=0.50)
        bow = vectorizer.fit_transform(data[analysis_type])

        # word importance plotting
        # words_freq = list(zip(vectorizer.get_feature_names_out(), [x[0] for x in bow.sum(axis=0).T.tolist()]))
        # words_freq = np.array(sorted(words_freq, key=lambda x: x[1], reverse=True))

        # plot_word_counts(output_folder, folder_name, words_freq, 20)

        # LSA
        lsa = TruncatedSVD(n_components=number_of_topics)
        lsa_matrix = lsa.fit_transform(bow)
        lsa_target = np.argmax(lsa_matrix, axis=1).tolist()
        lsa_target_percentages = np.amax(lsa_matrix, axis=1).tolist()

        # TSNE perplexity 20 orig
        tsne = TSNE(n_components=3, perplexity=25, learning_rate=100, n_iter=2000, random_state=0, angle=0.75)
        tsne_mat = tsne.fit_transform(lsa_matrix)

        # Cosine similarity
        cos_sim = cosine_similarity(lsa_matrix, lsa_matrix)
        lsa_top_three = {}

        for i, x in enumerate(cos_sim):
            # print(i, x)
            repo_name = repos[i]
            if repo_name not in queries:
                continue
            top_three_index = np.argpartition(x, -2)[-2:]
            top_three_values = x[top_three_index.astype(int)]
            top_three_repos = repos[top_three_index.astype(int)]
            sorted_top_five = [x for _, x in sorted(zip(top_three_values, top_three_repos), reverse=True)][-1:]
            for k, y in enumerate(sorted_top_five):
                lsa_top_three.setdefault("Query", []).append(repo_name)
                lsa_top_three.setdefault("Result", []).append(y)
                repo_url = '=HYPERLINK("https://github.com/'
                repo_url_owner = str(repo_name).split(" - ")[0]
                repo_url_name = str(repo_name).split(" - ")[1]
                y_repo_url_owner = str(y).split(" - ")[0]
                y_repo_url_name = str(y).split(" - ")[1]
                lsa_combined.setdefault("Query", []).append(repo_url + repo_url_owner + "/" + repo_url_name + '", "' + repo_name + '")')
                lsa_combined.setdefault("Result", []).append(repo_url + y_repo_url_owner + "/" + y_repo_url_name + '", "' + y + '")')
                lsa_combined.setdefault("Type", []).append(analysis_type)

        df = pd.DataFrame.from_dict(lsa_top_three)
        path = os.path.join(output_folder, folder_name + '/Query results.xlsx')
        df.to_excel(path, index=False)

        # Plotting setup
        fig = plt.figure(layout="constrained")
        fig.set_size_inches((18.5, 10.5), forward=False)
        gs = GridSpec(1, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0], projection='3d')
        ax2 = fig.add_subplot(gs[0, 1], projection='3d')

        # Plot
        colors = ListedColormap(sns.color_palette("hls", number_of_topics))
        lookup_table, color_target = np.unique(target, return_inverse=True)
        ax1.scatter(xs=tsne_mat[:, 0], ys=tsne_mat[:, 1], zs=tsne_mat[:, 2], c=color_target, cmap=colors)
        ax1.set_title("t-SNE with manual labels")

        # Plot axes 2
        ax2.scatter(xs=tsne_mat[:, 0], ys=tsne_mat[:, 1], zs=tsne_mat[:, 2], c=lsa_target, cmap=colors)
        ax2.set_title("t-SNE with LSA labels")

        # Export
        path = os.path.join(output_folder, folder_name + '/t-SNE Plot.png')
        plt.savefig(path, dpi=200)
        plt.close(fig)

        df = pd.DataFrame(lsa_matrix, columns=["Topic " + str(x) for x in range(number_of_topics)])
        df.insert(0, "Repository", repositories.data.keys())
        df.insert(1, "Actual target", target)
        df.insert(2, "Predicted target", lsa_target)
        path = os.path.join(output_folder, folder_name + '/LSA Output.xlsx')
        df.to_excel(path, index=False)

    df = pd.DataFrame.from_dict(lsa_combined)
    path = os.path.join(output_folder, 'Combined results.xlsx')
    df.to_excel(path, index=False)


if __name__ == "__main__":
    main()
