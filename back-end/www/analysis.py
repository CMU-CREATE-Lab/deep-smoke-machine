import sys
import pandas as pd
from split_metadata import *
from util import *
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, OrderedDict
from matplotlib import gridspec
from copy import deepcopy


def main(argv):
    vm = load_json("../data/metadata.json")
    df = pd.DataFrame.from_dict(vm)
    print("Number of videos: %d" % len(df))

    # Print label types
    df["label_type"] = df.apply(get_label_type, axis=1)
    for name, g in df.groupby(["label_type"]):
        print(name)
        print(g[["label_type", "label_state_admin", "label_state"]])

    # Aggregate labels
    df["label"] = df.apply(aggregate_label, axis=1)

    # Add datetime
    df["start_time"] = pd.to_datetime(df["start_time"], unit="s").dt.tz_localize("UTC").dt.tz_convert("US/Eastern")

    # Groups
    gp = df.groupby(["camera_id", "view_id"])

    # Plot dataset
    print("\n")
    print("="*40)
    print("== page 1")
    print("="*40)
    plot_dataset(gp, "../data/analysis/dataset_1.png",
            keys=[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)])
    print("\n")
    print("="*40)
    print("== page 2")
    print("="*40)
    plot_dataset(gp, "../data/analysis/dataset_2.png",
            keys=[(0, 5), (0, 6), (0, 7), (0, 8), (0, 9)])
    print("\n")
    print("="*40)
    print("== page 3")
    print("="*40)
    plot_dataset(gp, "../data/analysis/dataset_3.png",
            keys=[(0, 10), (0, 11), (0, 12), (0, 13), (0, 14)])
    print("\n")
    print("="*40)
    print("== page 4")
    print("="*40)
    plot_dataset(gp, "../data/analysis/dataset_4.png",
            keys=[(1, 0), (2, 0), (2, 1), (2, 2)])

    # Plot all
    df["dummy1"] = 0
    df["dummy2"] = 0
    print("\n")
    print("="*40)
    print("== all data")
    print("="*40)
    plot_dataset(df.groupby(["dummy1", "dummy2"]), "../data/analysis/dataset.png", use_ylim=False)


def plot_dataset(gp, p_out, p_frame="../data/rgb/", keys=None, use_ylim=True):
    gp = deepcopy(gp)
    print(gp.groups.keys())
    check_and_create_dir(p_out)

    n_rows = len(gp.groups.keys())
    if keys is not None:
        n_rows = len(keys)
    n_rows += 1
    width_ratios = [1, 0.6, 2.15, 1.5]
    n_cols = len(width_ratios)
    height_ratios = [0.02]+[1]*(n_rows-1)
    w = 2
    h = 2
    c = 0
    tick_font_size = 14

    fig = plt.figure(figsize=(w*sum(width_ratios), h*sum(height_ratios)))
    gs = gridspec.GridSpec(n_rows, n_cols, width_ratios=width_ratios, height_ratios=height_ratios)

    # Plot titles
    titles = ["View", "Label", "Season", "Time"]
    for t in titles:
        plt.subplot(gs[c])
        plt.text(0.5, 0.5, t, horizontalalignment="center", verticalalignment="center", fontsize=20)
        plt.axis("off")
        c += 1

    # Plot graphs
    for name, df in gp:
        if keys is not None and name not in keys: continue
        print("-"*20)
        print(name)
        # Randomly sample a file and plot it
        frame = []
        for fn in np.random.choice(df["file_name"], 1):
            fr = np.load(p_frame + fn + ".npy")
            frame.append(fr[np.random.choice(fr.shape[0]), ...])
        frame = np.concatenate(frame, axis=1)
        view_id = ("%d-%d" % (name[0], name[1]))
        plt.subplot(gs[c])
        plt.imshow(frame, aspect="equal")
        plt.text(90, 210, view_id, horizontalalignment="center", verticalalignment="center", fontsize=tick_font_size+8)
        plt.axis("off")
        c += 1
        # Plot label distribution
        L = OrderedDict(sorted(Counter(df["label"]).items()))
        print("$ label distribution: [yes, no]")
        print(L.values())
        ax = plt.subplot(gs[c])
        plt.bar(["no", "yes"], L.values(), alpha=0.6, width=0.65, color="black")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        plt.xticks(fontsize=tick_font_size)
        plt.autoscale(enable=True, axis="both", tight=True)
        if use_ylim:
            plt.yticks([0, 350, 700],fontsize=tick_font_size)
            plt.ylim((0, 700))
        else:
            plt.yticks(fontsize=tick_font_size)
        plt.margins(0.15, 0)
        plt.axis("on")
        c += 1
        # Plot distribution of season
        df["month"] = df.apply(to_month, axis=1)
        M = Counter(df["month"])
        for i in range(4): M[i] += 0
        M = OrderedDict(sorted(M.items()))
        print("$ season distribution: [winter, spring, summer, fall]")
        print(M.values())
        ax = plt.subplot(gs[c])
        plt.bar(["winter", "spring", "summer", "fall"], M.values(), alpha=0.6, width=0.35, color="black")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        plt.xticks(fontsize=tick_font_size)
        plt.autoscale(enable=True, axis="both", tight=True)
        if use_ylim:
            plt.yticks([0, 200, 400], fontsize=tick_font_size)
            plt.ylim((0, 400))
        else:
            plt.yticks(fontsize=tick_font_size)
        plt.margins(0.07, 0)
        c += 1
        # Plot distribution of time
        df["time"] = df.apply(to_time, axis=1)
        T = Counter(df["time"])
        for i in range(3): T[i] += 0
        T = OrderedDict(sorted(T.items()))
        print("$ time distribution: [6-10, 11-15, 16-20]")
        print(T.values())
        ax = plt.subplot(gs[c])
        plt.bar(["6-10", "11-15", "16-20"], T.values(), alpha=0.6, width=0.4, color="black")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        plt.xticks(fontsize=tick_font_size)
        plt.autoscale(enable=True, axis="both", tight=True)
        if use_ylim:
            plt.yticks([0, 250, 500], fontsize=tick_font_size)
            plt.ylim((0, 500))
        else:
            plt.yticks(fontsize=tick_font_size)
        plt.margins(0.1, 0)
        c += 1

    fig.tight_layout()
    plt.subplots_adjust(wspace=0.4, hspace=0.5)
    fig.savefig(p_out, dpi=100)
    fig.clf()
    plt.close()


def to_month(row):
    m = row["start_time"].month
    if m in [12, 1, 2]:
        return 0 # winter
    elif m in [3, 4, 5]:
        return 1 # spring
    elif m in [6, 7, 8]:
        return 2 # summer
    elif m in [9, 10, 11]:
        return 3 # fall
    else:
        return None


def to_time(row):
    t = row["start_time"].hour
    if t in [6, 7, 8, 9, 10]:
        return 0 # 6-10am
    elif t in [11, 12, 13, 14, 15]:
        return 1 # 11am-3pm
    elif t in [16, 17, 18, 19, 20]:
        return 2 # "4-8pm"
    else:
        return None


def aggregate_label(row):
    label_state_admin = row["label_state_admin"]
    label_state = row["label_state"]
    label = None
    has_error = False
    if label_state_admin == 47: # pos (gold standard)
        label = 1
    elif label_state_admin == 32: # neg (gold standard)
        label = 0
    elif label_state_admin == 23: # strong pos
        label = 1
    elif label_state_admin == 16: # strong neg
        label = 0
    else: # not determined by researchers
        if label_state == 23: # strong pos
            label = 1
        elif label_state == 16: # strong neg
            label = 0
        elif label_state == 20: # weak neg
            label = 0
        elif label_state == 19: # weak pos
            label = 1
        else:
            has_error = True
    if has_error:
        print("Error when aggregating label:")
        print(row)
    return label


def get_label_type(row):
    label_state_admin = row["label_state_admin"]
    label_state = row["label_state"]
    label_type = None
    has_error = False
    if label_state_admin in [47, 32]: # gold standards
        label_type = -1
    elif label_state_admin in [23, 16]: # researcher labels
        if label_state == -1: # citizen did not label
            label_type = 0 # only reseacher
        else:
            label_type = 1 # citizen-researcher collaboration
    else: # not determined by researchers
        if label_state in [23, 16, 20, 19]:
            label_type = 2 # only citizen
        else:
            has_error = True
    if has_error:
        print("Error when aggregating label:")
        print(row)
    return label_type


if __name__ == "__main__":
    main(sys.argv)
