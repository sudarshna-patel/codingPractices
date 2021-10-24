# load libraries
import numpy as np
import pandas as pd
import networkx as nx
import re
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm as tq

from tqdm.notebook import tqdm
tqdm.pandas()

df = pd.read_csv('Final_EL-small.csv', sep=',')

print("==================Undirected graph starts from here========================")
G = nx.Graph()
df_dict = df.to_dict('records')
# iterating over all rows and updating the directed graph
for row in tq(df_dict):
  G.add_edge(row['Source'], row['Target'], weight=row['Weight'], timestamp='Timestamp')

# 3.2.6, extracting giant component and calculate distance distibution
Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
GC = G.subgraph(Gcc[0])

# distance distribution in a graph
paths = pd.DataFrame(nx.all_pairs_shortest_path_length(w_subg))

l = paths[1].progress_apply(lambda x: list(x.values()))
ll = l.tolist()
flat_list = np.array([j for sub in ll for j in sub])
unique, counts = np.unique(flat_list, return_counts=True)

x = list(unique)
y = list(counts)
plt.loglog(x,y)
plt.title('Distribution')
plt.ylabel('Distance Distribution')
plt.xlabel('Distance')
plt.savefig('UD-DD-Small')
