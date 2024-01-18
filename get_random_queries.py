import os
import random
import sys

import configuration

configuration.load_properties(sys.argv[1])

all_repos = []

for folder in os.scandir(configuration.config["output-folder"].data):
    all_repos.append(folder.name)

with open("queries.txt", "w") as f:
    random_queries = random.sample(all_repos, 20)
    f.writelines(line + '\n' for line in random_queries)
