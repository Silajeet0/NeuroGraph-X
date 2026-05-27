"""
visualize.py  –  Readable visualization of A_spatial_master.pt
================================================================
Key improvements over the original:
  • Spectral layout clusters spatially close brain regions together,
    reflecting actual anatomical proximity rather than a circular ring.
  • Edges are colour-coded and thickness-scaled by RBF weight so
    strong connections stand out clearly.
  • Every node is labelled with its region index.
  • A colour-bar legend maps edge colour to edge weight.
  • High-weight edges are drawn on top of low-weight ones so they
    are never buried beneath lighter connections.
  • Optional: pass --threshold <value> (0-1) on the command line to
    hide edges below that weight and reduce clutter.
"""

import sys
import argparse
import torch
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Visualise A_spatial_master.pt")
parser.add_argument(
    "--pt", default="A_spatial_master.pt",
    help="Path to the .pt file (default: A_spatial_master.pt)"
)
parser.add_argument(
    "--threshold", type=float, default=0.0,
    help="Hide edges whose RBF weight is below this value (0–1). "
         "Try 0.3 or 0.5 for a much cleaner graph."
)
parser.add_argument(
    "--layout", choices=["spectral", "spring", "kamada"], default="spectral",
    help="Graph layout algorithm (default: spectral)."
)
parser.add_argument(
    "--output", default=None,
    help="Save the figure to this file path instead of showing it."
)
args = parser.parse_args()

# ── Load .pt ─────────────────────────────────────────────────────────────────
print(f"Loading {args.pt} …")
spatial_data = torch.load(args.pt, map_location="cpu")

NUM_NODES = 166

if isinstance(spatial_data, dict):
    print(f"Keys found: {list(spatial_data.keys())}")
    if "edge_index" in spatial_data:
        edge_index = spatial_data["edge_index"]
        edge_attr  = spatial_data.get("edge_attr", spatial_data.get("edge_weight", None))
    else:
        first_key   = list(spatial_data.keys())[0]
        tensor_data = spatial_data[first_key]
        if torch.is_tensor(tensor_data) and tensor_data.dim() == 2:
            edge_index = tensor_data.nonzero(as_tuple=False).t()
            edge_attr  = tensor_data[edge_index[0], edge_index[1]]
        else:
            raise ValueError(f"Cannot parse key '{first_key}'")
    pyg_spatial = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=NUM_NODES)

elif torch.is_tensor(spatial_data) and spatial_data.dim() == 2:
    edge_index  = spatial_data.nonzero(as_tuple=False).t()
    edge_attr   = spatial_data[edge_index[0], edge_index[1]]
    pyg_spatial = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=NUM_NODES)

elif hasattr(spatial_data, "edge_index"):
    pyg_spatial = spatial_data

else:
    raise TypeError(f"Unrecognised data type: {type(spatial_data)}")

# ── Build NetworkX graph ──────────────────────────────────────────────────────
print("Converting to NetworkX …")
G_full = to_networkx(pyg_spatial, to_undirected=True, edge_attrs=["edge_attr"])

# Normalise weights to [0, 1]
all_weights = np.array([d.get("edge_attr", 1.0) for _, _, d in G_full.edges(data=True)])
w_min, w_max = all_weights.min(), all_weights.max()
def normalise(w):
    return (w - w_min) / (w_max - w_min + 1e-9)

# Apply threshold filter
if args.threshold > 0:
    edges_to_remove = [
        (u, v) for u, v, d in G_full.edges(data=True)
        if normalise(d.get("edge_attr", 0.0)) < args.threshold
    ]
    G_full.remove_edges_from(edges_to_remove)
    print(f"After threshold={args.threshold:.2f}: {G_full.number_of_edges()} edges remain "
          f"(removed {len(edges_to_remove)})")
else:
    print(f"Total edges: {G_full.number_of_edges()}")

# ── Layout ────────────────────────────────────────────────────────────────────
print(f"Computing '{args.layout}' layout …")
if args.layout == "spectral":
    pos = nx.spectral_layout(G_full, weight="edge_attr")
elif args.layout == "spring":
    pos = nx.spring_layout(G_full, weight="edge_attr", seed=42, iterations=100)
else:  # kamada
    pos = nx.kamada_kawai_layout(G_full, weight="edge_attr")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(20, 20), facecolor="#0d0d1a")
ax.set_facecolor("#0d0d1a")
ax.axis("off")

# Colourmap: low weight = cool blue, high weight = warm amber/red
cmap = cm.plasma

# Sort edges by weight so heavy ones are drawn on top
edge_data = sorted(
    G_full.edges(data=True),
    key=lambda e: normalise(e[2].get("edge_attr", 0.0))
)

# Draw edges in batches by weight quintile for performance
edge_list   = [(u, v) for u, v, _ in edge_data]
edge_w_norm = np.array([normalise(d.get("edge_attr", 0.0)) for _, _, d in edge_data])
edge_colors = [cmap(w) for w in edge_w_norm]
edge_widths = 0.3 + edge_w_norm * 3.5   # thin for weak, thick for strong

nx.draw_networkx_edges(
    G_full, pos,
    edgelist=edge_list,
    width=edge_widths,
    edge_color=edge_colors,
    alpha=0.75,
    ax=ax
)

# ── Nodes ─────────────────────────────────────────────────────────────────────
# Colour nodes by degree (connectivity)
degrees    = dict(G_full.degree())
deg_values = np.array([degrees[n] for n in G_full.nodes()])
deg_norm_min = deg_values.min()
deg_norm   = (deg_values - deg_norm_min) / (deg_values.max() - deg_norm_min + 1e-9)
node_colors = [cm.cool(d) for d in deg_norm]

nx.draw_networkx_nodes(
    G_full, pos,
    node_size=120,
    node_color=node_colors,
    edgecolors="white",
    linewidths=0.4,
    ax=ax
)

# ── Node labels (region index) ────────────────────────────────────────────────
# Draw all labels; use small font so they don't overlap too much
label_dict = {n: str(n) for n in G_full.nodes()}
nx.draw_networkx_labels(
    G_full, pos,
    labels=label_dict,
    font_size=5,
    font_color="white",
    font_weight="bold",
    ax=ax
)

# ── Colour-bars via inset_axes (never steals space from main axes) ────────────
# ── Colour-bars (Legends) ─────────────────────────────────────────────────────
# Using absolute positioning [left, bottom, width, height] to guarantee no overlap

# Edge-weight bar — top right
cax_edge = fig.add_axes([0.92, 0.55, 0.015, 0.35]) 
sm_edge = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=w_min, vmax=w_max))
sm_edge.set_array([])
cbar_edge = fig.colorbar(sm_edge, cax=cax_edge)
cbar_edge.set_label("RBF Kernel Weight", color="white", fontsize=11, labelpad=8) 
cbar_edge.ax.yaxis.set_tick_params(color="white", labelsize=9)
plt.setp(cbar_edge.ax.yaxis.get_ticklabels(), color="white")

# Node-degree bar — bottom right
cax_node = fig.add_axes([0.92, 0.15, 0.015, 0.35])
sm_node = plt.cm.ScalarMappable(cmap=cm.cool,
                                 norm=mcolors.Normalize(vmin=deg_values.min(),
                                                        vmax=deg_values.max()))
sm_node.set_array([])
cbar_node = fig.colorbar(sm_node, cax=cax_node)
cbar_node.set_label("Node Degree", color="white", fontsize=11, labelpad=8)
cbar_node.ax.yaxis.set_tick_params(color="white", labelsize=9)
plt.setp(cbar_node.ax.yaxis.get_ticklabels(), color="white")

# ── Titles and Footers ────────────────────────────────────────────────────────
fig.suptitle(
    f"Spatial Weighted Adjacency Graph",
    fontsize=18, fontweight="bold", color="white",
    x=0.45, y=0.98 # Shifted slightly left to balance the legends
)

note = (f"Layout: {args.layout}  |  "
        f"Nodes: {G_full.number_of_nodes()}  |  "
        f"Edges: {G_full.number_of_edges()}")
fig.text(0.45, 0.01, note, ha="center", fontsize=10, color="#aaaaaa")

# IMPORTANT FIX: The 0.88 here tells the main graph to stop at 88% of the width, 
# leaving the remaining 12% exclusively for our legends and their text labels.
plt.tight_layout(rect=[0, 0.03, 0.88, 0.96])

if args.output:
    plt.savefig(args.output, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {args.output}")
else:
    plt.show()
