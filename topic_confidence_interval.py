import math
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('analysis_3d/results.csv')
stats = df.groupby(['Type'])['Topic match'].agg(['mean', 'count', 'std'])
confidence = []

for i in stats.index:
    m, c, s = stats.loc[i]
    confidence.append(1.96*s/math.sqrt(c))

stats['confidence'] = confidence

print(stats)
stats.to_csv('analysis_3d/topic_match_stats.csv')

names_split = [" ".join(x.split("_")).capitalize() for x in list(stats.index.values)[1::2]]
names_non_split = [" ".join(x.split("_")).capitalize() for x in list(stats.index.values)[::2]]

split = stats['mean'][1::2]
non_split = stats['mean'][::2]

confidence_split = stats['confidence'][1::2]
confidence_non_split = stats['confidence'][::2]

barWidth = 0.3
r1 = np.arange(len(non_split))
r2 = [x + barWidth for x in r1]
f = plt.figure()
f.set_figwidth(12)
f.set_figheight(4)
plt.bar(r1, non_split, width = barWidth, color = 'black', edgecolor = 'black', yerr=confidence_non_split, capsize=7, alpha=0.5, label="Non-split")
plt.bar(r2, split, width = barWidth, color = 'gray', edgecolor = 'black', yerr=confidence_split, capsize=7,alpha=0.5, label="Split")
# general layout
plt.xticks([r + barWidth/2 for r in range(len(non_split))], names_non_split, fontsize=10)
plt.ylabel('Average')
plt.ylim(0, 1.10)
plt.xlabel('Type of identifier')
plt.legend()

# Show graphic
plt.title("Confidence intervals for finding a project with matching topic")
path = os.path.join('analysis_3d', 'topic_match_stats.png')
plt.savefig(path)
#plt.show()
