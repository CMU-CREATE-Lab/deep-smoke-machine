import sys
from collections import defaultdict
import json
import numpy as np
import copy
from util import *

#np.random.shuffle()

def divide_list(target_keys, frac_valid=0.1, frac_test=0.3):
    n_keys = len(target_keys)
    n_valid = int(n_keys*frac_valid)
    n_test = int(n_keys*frac_test)
    test_key = target_keys[:n_test]
    valid_key = target_keys[n_test:n_valid+n_test]
    train_key = target_keys[n_valid+n_test:]
    return (train_key, valid_key, test_key)


def split(vm_dict, target_key_type, train_key=None, valid_key=None, test_key=None):
    if train_key is None or valid_key is None or test_key is None:
        target_keys = list(vm_dict.keys())
        np.random.shuffle(target_keys)
        # 10% for validation, 30% for testing
        train_key, valid_key, test_key = divide_list(target_keys, frac_valid=0.1, frac_test=0.3)
    vm_valid = []
    vm_test = []
    vm_train = []
    for d in vm_dict:
        if d in valid_key:
            vm_valid += vm_dict[d]
        elif d in test_key:
            vm_test += vm_dict[d]
        elif d in train_key:
            vm_train += vm_dict[d]
    print("\nTraining:")
    print_distribution(vm_train, target_key_type=target_key_type)
    print("\nValidation:")
    print_distribution(vm_valid, target_key_type=target_key_type)
    print("\nTesting:")
    print_distribution(vm_test, target_key_type=target_key_type)
    n = len(vm_valid) + len(vm_test) + len(vm_train)
    print("\nSize of validation set: %d (%.2f)" % (len(vm_valid), len(vm_valid)/n))
    print("Size of test set: %d (%.2f)" % (len(vm_test), len(vm_test)/n))
    print("Size of training set: %d (%.2f)" % (len(vm_train), len(vm_train)/n))
    return (vm_train, vm_valid, vm_test)


def print_distribution(vm, target_key_type):
    count_vm = defaultdict(lambda: defaultdict(int))
    for v in vm:
        k = to_key(v, target_key_type)
        label = v["label"]
        if label == 1:
            count_vm[k]["pos"] += 1
        elif label == 0:
            count_vm[k]["neg"] += 1
    print(json.dumps(count_vm, indent=4))


def to_key(v, target_key_type):
    # The file name contains information about camera, date, and bounding box
    # We can use this as the key of the dataset for separating training, validation, and test sets
    key = v["file_name"].split("-")
    if target_key_type == "camera":
        return "-".join(key[0:2])
    elif target_key_type == "date":
        return "-".join(key[2:5])


# Aggregate labels from citizens (label_state) and researchers (label_state_admin)
# "label" means the final aggregated label
# "weight" means the confidence of the aggregated label
def aggregate_label(vm):
    vm = copy.deepcopy(vm)
    vm_new = []
    for i in range(len(vm)):
        has_error = False
        v = vm[i]
        label_state_admin = v.pop("label_state_admin", None)
        label_state = v.pop("label_state", None)
        if label_state_admin == 47: # pos (gold standard)
            v["label"] = 1
            v["weight"] = 1
        elif label_state_admin == 32: # neg (gold standard)
            v["label"] = 0
            v["weight"] = 1
        elif label_state_admin == 23: # strong pos
            v["label"] = 1
            if label_state == 23: # strong pos
                v["weight"] = 1 # (1+1)/2
            elif label_state == 16: # strong neg
                v["weight"] = 0.5 # (1+0)/2
            elif label_state == 20: # weak neg
                v["weight"] = 0.66 # (1+0.33)/2
            elif label_state == 19: # weak pos
                v["weight"] = 0.83 # (1+0.66)/2
            else: # not determined by citizens
                v["weight"] = 0.75
        elif label_state_admin == 16: # strong neg
            v["label"] = 0
            if label_state == 23: # strong pos
                v["weight"] = 0.5 # (1+0)/2
            elif label_state == 16: # strong neg
                v["weight"] = 1 # (1+1)/2
            elif label_state == 20: # weak neg
                v["weight"] = 0.83 # (1+0.66)/2
            elif label_state == 19: # weak pos
                v["weight"] = 0.66 # (1+0.33)/2
            else: # not determined by citizens
                v["weight"] = 0.75
        else: # not determined by researchers
            if label_state == 23: # strong pos
                v["label"] = 1
                v["weight"] = 1
            elif label_state == 16: # strong neg
                v["label"] = 0
                v["weight"] = 1
            elif label_state == 20: # weak neg
                v["label"] = 0
                v["weight"] = 0.66
            elif label_state == 19: # weak pos
                v["label"] = 1
                v["weight"] = 0.66
            else:
                has_error = True
        if has_error or "label" not in v or "weight" not in v:
            print("Error when aggregating label:")
            print(v)
        else:
            vm_new.append(v)

    return vm_new


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
            #if label_state_admin in defined and label_state in defined:
            if label_state in defined:
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
