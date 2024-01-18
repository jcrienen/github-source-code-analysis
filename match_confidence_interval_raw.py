import math
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('analysis_3d/results.csv')
stats = df.groupby(['Type','Score'])['Score'].agg(['size']).unstack(level=-1)

combined = {}

print(stats)
#stats.to_csv('analysis_3d/repository_match_stats_raw.csv')

# for value in stats.index.values:
#     combined.setdefault("analysis_type", []).append(value[0])
#     combined.setdefault("score", []).append(value[1])
#
# for value in stats['size']:
#     combined.setdefault("size", []).append(value)

#print(stats)

names_split = [" ".join(x.split("_")).capitalize() for x in list(stats.index.values)[1::2]]
names_non_split = [" ".join(x.split("_")).capitalize() for x in list(stats.index.values)[::2]]

names = [" ".join(x.split("_")).capitalize() for x in list(stats.index.values)]

weights = {}
for i, analysis_type in enumerate(list(stats.index.values)):
    print(analysis_type)
    weights.setdefault("1", []).append(stats.values[i][0])
    weights.setdefault("2", []).append(stats.values[i][1])
    weights.setdefault("3", []).append(stats.values[i][2])
    weights.setdefault("4", []).append(stats.values[i][3])

weights_split = {}
weights_non_split = {}

weights_split.setdefault("1", weights["1"][1::2])
weights_split.setdefault("2", weights["2"][1::2])
weights_split.setdefault("3", weights["3"][1::2])
weights_split.setdefault("4", weights["4"][1::2])

weights_non_split.setdefault("1", weights["1"][::2])
weights_non_split.setdefault("2", weights["2"][::2])
weights_non_split.setdefault("3", weights["3"][::2])
weights_non_split.setdefault("4", weights["4"][::2])

barWidth = 0.3
r1 = np.arange(len(names_non_split))

r2 = [x + barWidth for x in r1]
f = plt.figure()
f.set_figwidth(12)
f.set_figheight(4)

bottom = np.zeros(6)
for boolean, weight_count in weights_non_split.items():
    plt.bar(r1, weight_count, width=barWidth, color='black', edgecolor='black', alpha=0.5, label="Non-split" if boolean == "1" else '', bottom=bottom)
    bottom += weight_count

bottom = np.zeros(6)
for boolean, weight_count in weights_split.items():
    plt.bar(r2, weight_count, width=barWidth, color='gray', edgecolor='black', label="Split" if boolean == "1" else '', alpha=0.5,bottom=bottom)
    bottom += weight_count

# # general layout

#plt.bar(r1, non_split, width = barWidth, color = 'black', edgecolor = 'black', yerr=confidence_non_split, capsize=7, alpha=0.5, label="Non-split")
#plt.bar(r2, split, width = barWidth, color = 'gray', edgecolor = 'black', yerr=confidence_split, capsize=7,alpha=0.5, label="Split")

plt.xticks([r + barWidth/2 for r in range(len(names_non_split))], names_non_split, fontsize=10)
plt.ylabel('Occurrence')
plt.ylim(0, 20)
plt.xlabel('Type of identifier')
plt.legend()
#
# # Show graphic
plt.title("Manual scores for each type of identifiers")
# path = os.path.join('analysis_3d', 'repository_match_stats_raw.png')
# plt.savefig(path)
plt.show()
