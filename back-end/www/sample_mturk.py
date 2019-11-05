import sys
from collections import defaultdict
import json
import numpy as np
import copy
from util import *


# Select only the labels with both citizen and researcher labels (no gold standards)
# Also select the gold standards
def filter_labels(vm):
    gold_pos = []
    gold_neg = []
    selected = []
    for v in vm:
        label_state = v["label_state"]
        label_state_admin = v["label_state_admin"]
        if label_state_admin == 47: # pos (gold standard)
            gold_pos.append(v)
        elif label_state_admin == 32: # neg (gold standard)
            gold_neg.append(v)
        else:
            defined = [23, 16, 20, 19]
            if label_state_admin in defined and label_state in defined:
                selected.append(v)
    return (selected, gold_pos, gold_neg)


# Split metadata into training, validation, and test sets
def main(argv):
    # Check
    if len(argv) > 1:
        if argv[1] != "confirm":
            print("Must confirm by running: python sample_mturk.py confirm")
            return
    else:
        print("Must confirm by running: python sample_mturk.py confirm")
        return

    # Load and choose data
    vm = load_json("../data/metadata.json")
    selected, gold_pos, gold_neg = filter_labels(vm)
    np.random.shuffle(selected) # random shuffle
    n = 60 # number of batches
    selected = selected[:n*12] # choose a subset for mturk to label
    print("Select %d samples" % (len(selected)))

    # Form batches
    selected = np.array(selected)
    selected = np.split(selected, n)
    gold_pos = np.array(gold_pos)
    gold_neg = np.array(gold_neg)
    batch_wa = []
    for i in range(n):
        a = [(1,3), (2,2), (3,1)]
        a = a[np.random.choice(len(a))]
        gp = gold_pos[np.random.choice(len(gold_pos), a[0])]
        gn = gold_neg[np.random.choice(len(gold_neg), a[1])]
        v = list(selected[i]) + list(gp) + list(gn)
        np.random.shuffle(v)
        batch_wa.append(v)

    # Remove label states
    batch = []
    for b_wa in batch_wa:
        b = []
        for v_wa in b_wa:
            v = {}
            v["id"] = v_wa["id"]
            v["url_part"] = v_wa["url_part"]
            v["url_root"] = v_wa["url_root"]
            b.append(v)
        batch.append(b)
    p_save = "../data/mturk_batch.json"
    p_save_wa = "../data/mturk_batch_with_answers.json"
    save_json(batch, p_save)
    save_json(batch_wa, p_save_wa)
    print("The mturk batch is saved in: " + p_save)
    print("The mturk batch (with answers) is saved in: " + p_save_wa)


if __name__ == "__main__":
    main(sys.argv)
