import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
import plotly.graph_objects as go
import time
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
from tqdm import tqdm
import community
import collections



def plot_actor_network(G, k=0.2, seed=42, enable_partition=False):
    """
    Plot the actor network using spring layout and optionally cluster with the Louvain method.
    
    """
    start_time = time.time()

    values = None
    if enable_partition:
        partition = community.best_partition(G, random_state=42)
        values = [partition.get(node) for node in G.nodes()]
        counter = collections.Counter(values)
        print(f"Partition summary: {counter}")

    sp = nx.spring_layout(G, k=k, seed=seed)
    plt.figure(figsize=(15, 15))
    nx.draw_networkx(
        G,
        pos=sp,
        with_labels=False,
        node_size=5,
        node_color=values if values is not None else '#1f78b4',
        width=0.05,
    )
    plt.title("Actor network clustered with Louvain method" if enable_partition else "Actor network", fontsize=15)
    plt.show()

    end_time = time.time()
    print(f"Time to compute: {end_time - start_time:.1f} seconds") 