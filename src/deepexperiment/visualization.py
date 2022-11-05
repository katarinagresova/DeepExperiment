import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def plot_t(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+0.4, base],
                  width=0.2, height=height, facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge, base+0.85*height],
                  width=1.0, height=0.15*height, facecolor=color, edgecolor=color, fill=True))
    
def plot_g(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=1.3, height=height,
                                            facecolor=color, edgecolor=color))
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=0.7*1.3, height=0.7*height,
                                            facecolor='white', edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+1, base], width=1.0, height=height,
                                            facecolor='white', edgecolor='white', fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+0.825, base+0.085*height], width=0.174, height=0.415*height,
                                            facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+0.625, base+0.35*height], width=0.374, height=0.15*height,
                                            facecolor=color, edgecolor=color, fill=True))
    
def plot_a(ax, base, left_edge, height, color):
    a_polygon_coords = [
        np.array([
           [0.0, 0.0],
           [0.5, 1.0],
           [0.5, 0.8],
           [0.2, 0.0],
        ]),
        np.array([
           [1.0, 0.0],
           [0.5, 1.0],
           [0.5, 0.8],
           [0.8, 0.0],
        ]),
        np.array([
           [0.225, 0.45],
           [0.775, 0.45],
           [0.85, 0.3],
           [0.15, 0.3],
        ])
    ]
    for polygon_coords in a_polygon_coords:
        ax.add_patch(matplotlib.patches.Polygon((np.array([1,height])[None,:]*polygon_coords
                                                 + np.array([left_edge,base])[None,:]),
                                                facecolor=color, edgecolor=color))


def plot_c(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=1.3, height=height,
                                            facecolor=color, edgecolor=color))
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=0.7*1.3, height=0.7*height,
                                            facecolor='white', edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+1, base], width=1.0, height=height,
                                            facecolor='white', edgecolor='white', fill=True))
    
def plot_dash(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+0.2, base+0.3*height], width=0.6, height=0.2*height,
                                            facecolor=color, edgecolor=color, fill=True))
    
def plot_line(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+0.45, base+0.2*height], width=0.1, height=0.6*height,
                                            facecolor=color, edgecolor=color, fill=True))
    
def plot_dot(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.5, base+0.5*height], width=0.3, height=0.2*height,
                                            facecolor=color, edgecolor=color))
    
def plot_block(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge, base], width=1, height=height,
                                            facecolor=color, edgecolor=color, fill=True))

color_palette = ['#034da1', '#3359a7', '#4c66ad', '#6174b3', '#7482b9', '#8590be', '#979ec4', '#a8adca', '#b9bbcf', '#c9cbd5', '#dadada', '#dac4c4', '#daaeae', '#da9999', '#da8383', '#da6d6d', '#d95757', '#d94141', '#d92c2c', '#d91616', '#d90000']

plot_base = {
    "A": plot_a,
    "C": plot_c,
    "T": plot_t,
    "G": plot_g,
    "-": plot_dash
}

complement = {
    "A": "T",
    "T": "A",
    "C": "G",
    "G": "C",
    "-": ""
}

def plot_alignment(align_x, align_s, align_y, arrows = True):
    fig = plt.figure(figsize=(20,2))
    ax = fig.add_subplot(111) 

    max_absolute_value = max(abs(np.array(align_s)))
    s = align_s / max_absolute_value
    s = s * 10 + 10

    length_padding = 0.2
    height_padding = 0.2

    ax.set_xlim(-length_padding, len(align_x) * (1 + length_padding))
    ax.set_ylim(0, 4 * (1 + height_padding))

    if arrows:
        ax.add_patch(matplotlib.patches.Arrow(x=0, y=0.2, dx=2, dy=0, width=0.6, color=color_palette[10]))

    for i in range(len(align_x)):

        plot_base[align_y[i]](ax, 0.6, 0 + (1.2 * i), 1, color_palette[round(s[i])])
        if align_y[i] == complement[align_x[i]]:
            plot_line(ax, 1.7, 0 + (1.2 * i), 1, color_palette[round(s[i])])
        elif align_y[i] in "ACTG" and align_x[i] in "ACTG":
            plot_dot(ax, 1.7, 0 + (1.2 * i), 1, color_palette[round(s[i])])
        plot_base[align_x[i]](ax, 3.0, 0 + (1.2 * i), 1, color_palette[round(s[i])])

    if arrows:
        ax.add_patch(matplotlib.patches.Arrow(x=(1.2 * len(align_x)) - length_padding, y=4.4, dx=-2, dy=0, width=0.6, color=color_palette[10]))

    plt.axis('off')
    plt.plot();

def plot_seq_agn_alignment(align_x, align_s, align_y):
    fig = plt.figure(figsize=(20,0.5))
    ax = fig.add_subplot(111) 

    max_absolute_value = max(abs(np.array(align_s)))
    s = align_s / max_absolute_value
    s = s * 10 + 10

    length_padding = 0.2

    ax.set_xlim(-length_padding, len(align_x) * (1 + length_padding))
    ax.set_ylim(0, 1)

    for i in range(len(align_x)):

        if align_y[i] == complement[align_x[i]]:
            plot_line(ax, 0, 0 + (1.2 * i), 1, color_palette[round(s[i])])
        elif align_y[i] in "ACTG" and align_x[i] in "ACTG":
            plot_dot(ax, 0, 0 + (1.2 * i), 1, color_palette[round(s[i])])
        
    plt.axis('off')
    plt.plot();

def plot_gene_importance(align_s):

    # reverse order
    align_s = align_s[::-1]

    fig = plt.figure(figsize=(20,0.5))
    ax = fig.add_subplot(111) 

    max_absolute_value = max(abs(np.array(align_s)))
    s = align_s / max_absolute_value
    s = s * 10 + 10

    shift = 0.5

    ax.set_xlim(1 - shift, len(s) - shift + 1)
    ax.set_ylim(0, 1)

    for i in range(len(s)):
        plot_block(ax, 0, 1 - shift + (1 * i), 1, color_palette[round(s[i])])

    # Turn off ticks and labels
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_xticklabels(np.arange(1, len(align_s) + 1))
    ax.set_xticks(np.arange(1, len(align_s) + 1))
    plt.plot();

def plotbar_gene_importance(align_x, align_s):

    # reverse order
    align_s = align_s[::-1]
    align_x = align_x[::-1]

    fig, ax = plt.subplots(figsize = (20,4))

    max_absolute_value = max(abs(np.array(align_s)))
    s = align_s / max_absolute_value
    s = s * 10 + 10
    mRNA_s = []
    counter = 1
    for i in range(len(align_x)):
        if align_x[i] != "-":
            mRNA_s.append(align_s[i])
            ax.bar(x=counter, height=align_s[i], color=color_palette[round(s[i])])
            counter += 1

    ax.set_xticklabels(np.arange(1, len(mRNA_s) + 1))
    ax.set_xticks(np.arange(1, len(mRNA_s) + 1))
    ax.plot();

def plot_miRNA_importance(align_y, align_s):
    fig = plt.figure(figsize=(10,0.5))
    ax = fig.add_subplot(111) 

    max_absolute_value = max(abs(np.array(align_s)))
    s = align_s / max_absolute_value
    s = s * 10 + 10
    
    shift = 0.5

    miRNA = []
    miRNA_s = []
    for i in range(len(align_y)):
        if align_y[i] != "-":
            miRNA.append(align_y[i])
            miRNA_s.append(s[i])

    ax.set_xlim(1-shift, len(miRNA)-shift+1)
    ax.set_ylim(0, 1)

    for i in range(len(miRNA)):
        plot_block(ax, 0, 1-shift + (1 * i), 1, color_palette[round(miRNA_s[i])])

    # Turn off ticks and labels
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_xticklabels(np.arange(1, len(miRNA) + 1))
    ax.set_xticks(np.arange(1, len(miRNA) + 1))
    plt.plot();

def plotbar_miRNA_importance(align_y, align_s):

    fig, ax = plt.subplots(figsize = (10,4))

    max_absolute_value = max(abs(np.array(align_s)))
    s = align_s / max_absolute_value
    s = s * 10 + 10
    miRNA_s = []
    counter = 1
    for i in range(len(align_y)):
        if align_y[i] != "-":
            miRNA_s.append(align_s[i])
            ax.bar(x=counter, height=align_s[i], color=color_palette[round(s[i])])
            counter += 1

    ax.set_xticklabels(np.arange(1, len(miRNA_s) + 1))
    ax.set_xticks(np.arange(1, len(miRNA_s) + 1))
    ax.plot();

def plot_miRNA_importance_w_spaces(align_y, align_s):
    
    fig = plt.figure(figsize=(15,0.5))
    ax = fig.add_subplot(111) 

    # normalize score to the range <-1, 1>
    max_absolute_value = max(abs(np.array(align_s)))
    s = align_s / max_absolute_value
    # normalize score to the range <0, 20>
    s = s * 10 + 10

    shift = 0.5

    start_i = min(align_y.index('A'), align_y.index('T'), align_y.index('C'), align_y.index('G'))
    # reverse order
    align_y = align_y[::-1]     
    end_i = len(align_y) - min(align_y.index('A'), align_y.index('T'), align_y.index('C'), align_y.index('G'))
    #reverse back
    align_y = align_y[::-1]  
    miRNA = align_y[start_i:end_i]
    miRNA_s = s[start_i:end_i]

    ax.set_xlim(1-shift, len(miRNA) - shift+1)
    ax.set_ylim(0, 1)

    for i in range(len(miRNA)):
        if miRNA[i] in "ACTG":
            plot_block(ax, 0, 1-shift + (1 * i), 1, color_palette[round(miRNA_s[i])])
        else:
            plot_block(ax, 0, 1-shift + (1 * i), 1, "white")

    # Turn off ticks and labels
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_xticklabels(np.arange(1, len(miRNA) + 1))
    ax.set_xticks(np.arange(1, len(miRNA) + 1))
    plt.plot();

def plotbar_miRNA_importance_w_spaces(align_y, align_s):

    start_i = min(align_y.index('A'), align_y.index('T'), align_y.index('C'), align_y.index('G'))
    # reverse order
    align_y = align_y[::-1]     
    end_i = len(align_y) - min(align_y.index('A'), align_y.index('T'), align_y.index('C'), align_y.index('G'))
    miRNA_s = align_s[start_i:end_i]

    fig, ax = plt.subplots(figsize = (10,4))

    max_absolute_value = max(abs(np.array(miRNA_s)))
    s = miRNA_s / max_absolute_value
    s = s * 10 + 10

    for i in range(0, len(miRNA_s)):
        ax.bar(x=i + 1, height=miRNA_s[i], color=color_palette[round(s[i])])

    ax.set_xticklabels(np.arange(1, len(miRNA_s) + 1))
    ax.set_xticks(np.arange(1, len(miRNA_s) + 1))
    ax.plot();

def plot_cluster(ax, y_cluster, s_cluster, cluster_consensus, title):

    max_absolute_value = max(abs(np.array(cluster_consensus)))
    s = np.array(cluster_consensus) / max_absolute_value
    s = s * 10 + 10

    for i in range(len(s)):
        ax.bar(x=i+1, height=cluster_consensus[i], color=color_palette[round(s[i])])

    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

    for i in range(50):
        index = np.random.randint(0, len(y_cluster))

        max_absolute_value = max(abs(np.array(s_cluster[index])))
        s = s_cluster[index] / max_absolute_value
        s = s * 10 + 10

        shift = 0.5
        height = 0.01

        miRNA = []
        miRNA_s = []
        for j in range(len(y_cluster[index])):
            if y_cluster[index][j] != "-":
                miRNA.append(y_cluster[index][j])
                miRNA_s.append(s[j])

        base = -1 * height * i - height
        for j in range(len(miRNA)):
            plot_block(ax, base, 1-shift + (1 * j), height, color_palette[round(miRNA_s[j])])


    ax.set_xlim(0.5, 20 + 0.5)
    ax.set_ylim(-0.5, 1)

    ax.set_xticklabels(np.arange(1, len(miRNA_s) + 1))
    ax.set_xticks(np.arange(1, len(miRNA_s) + 1))

    ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    ax.set_title(title)