import matplotlib.pyplot as plt
import numpy as np


def visualize_max_cut_concept():
    """Show what max cut means with a simple example"""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Example graph
    nodes = {0: (0, 1), 1: (1, 1), 2: (0, 0), 3: (1, 0)}
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (0, 3)]

    # Draw original graph
    ax = axes[0]
    for node, (x, y) in nodes.items():
        ax.scatter(x, y, s=800, c='lightblue', edgecolor='black', linewidth=2)
        ax.text(x, y, str(node), ha='center', va='center', fontsize=14, fontweight='bold')

    for edge in edges:
        u, v = edge
        x1, y1 = nodes[u]
        x2, y2 = nodes[v]
        ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2)

    ax.set_title('Original Graph\n(5 edges total)', fontsize=12, fontweight='bold')
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.axis('off')

    # Bad partition (cut = 3)
    ax = axes[1]
    set_s = [0, 1]  # Red nodes
    set_t = [2, 3]  # Blue nodes

    for node, (x, y) in nodes.items():
        color = 'lightcoral' if node in set_s else 'lightblue'
        ax.scatter(x, y, s=800, c=color, edgecolor='black', linewidth=2)
        ax.text(x, y, str(node), ha='center', va='center', fontsize=14, fontweight='bold')

    cut_edges = 0
    for edge in edges:
        u, v = edge
        x1, y1 = nodes[u]
        x2, y2 = nodes[v]

        if (u in set_s and v in set_t) or (u in set_t and v in set_s):
            ax.plot([x1, x2], [y1, y2], 'red', linewidth=3, label='Cut edge')
            cut_edges += 1
        else:
            ax.plot([x1, x2], [y1, y2], 'gray', linewidth=1, alpha=0.5)

    ax.set_title(f'Poor Cut: {cut_edges} edges\nRed group: {set_s}\nBlue group: {set_t}',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.axis('off')

    # Good partition (cut = 4)
    ax = axes[2]
    set_s = [1, 2]  # Red nodes
    set_t = [0, 3]  # Blue nodes

    for node, (x, y) in nodes.items():
        color = 'lightcoral' if node in set_s else 'lightblue'
        ax.scatter(x, y, s=800, c=color, edgecolor='black', linewidth=2)
        ax.text(x, y, str(node), ha='center', va='center', fontsize=14, fontweight='bold')

    cut_edges = 0
    for edge in edges:
        u, v = edge
        x1, y1 = nodes[u]
        x2, y2 = nodes[v]

        if (u in set_s and v in set_t) or (u in set_t and v in set_s):
            ax.plot([x1, x2], [y1, y2], 'red', linewidth=3)
            cut_edges += 1
        else:
            ax.plot([x1, x2], [y1, y2], 'gray', linewidth=1, alpha=0.5)

    ax.set_title(f'Better Cut: {cut_edges} edges\nRed group: {set_s}\nBlue group: {set_t}',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.axis('off')

    plt.tight_layout()
    plt.suptitle('Goal: Find the partition that cuts the MOST edges!',
                 fontsize=16, fontweight='bold', y=1.05)
    plt.show()

    print("💡 KEY IDEA: We want to split the graph into two groups so that")
    print("   as many edges as possible go BETWEEN the groups (not within groups)")


visualize_max_cut_concept()