import sys
from collections import defaultdict
import json
import numpy as np
from util import *

def split_and_save_data(vm, target_key_type, method="assign"):
    # Index metadata by date or camera
    vm_dict = defaultdict(list)
    for v in vm:
        k = to_key(v, target_key_type)
        vm_dict[k].append(v)

    p = "../data/split/"
    check_and_create_dir(p)
    print("="*40)
    print("="*40)
    print("Split data by " + target_key_type)
    if method == "random":
        vm_train, vm_valid, vm_test = split(vm_dict, target_key_type)
        save_json(vm_valid, p+"metadata_validation_random_split_by_"+target_key_type+".json")
        save_json(vm_test, p+"metadata_test_random_split_by_"+target_key_type+".json")
        save_json(vm_train, p+"metadata_train_random_split_by_"+target_key_type+".json")
    elif method == "assign":
        if target_key_type == "camera":
            three_splits = [
                {
                    "train": ["0-0", "0-3", "0-6", "0-8", "0-12", "0-2", "0-5", "0-9", "0-11", "0-14"],
                    "valid": ["0-4", "0-10"],
                    "test": ["1-0", "2-0", "2-1", "2-2", "0-1", "0-7", "0-13"]
                }, {
                    "train": ["0-1", "0-4", "0-7", "0-10", "0-13", "0-0", "0-3", "0-6", "0-8", "0-12"],
                    "valid": ["0-5", "0-11"],
                    "test": ["1-0", "2-0", "2-1", "2-2", "0-2", "0-9", "0-14"]
                }, {
                    "train": ["0-2", "0-5", "0-9", "0-11", "0-14", "0-1", "0-4", "0-7", "0-10", "0-13"],
                    "valid": ["0-3", "0-8"],
                    "test": ["1-0", "2-0", "2-1", "2-2", "0-0", "0-6", "0-12"]
                }]
            for i in range(len(three_splits)):
                s = three_splits[i]
                vm_train, vm_valid, vm_test = split(vm_dict, target_key_type,
                        train_key=s["train"], valid_key=s["valid"], test_key=s["test"])
                save_json(vm_valid, p+"metadata_validation_split_"+str(i)+"_by_"+target_key_type+".json")
                save_json(vm_test, p+"metadata_test_split_"+str(i)+"_by_"+target_key_type+".json")
                save_json(vm_train, p+"metadata_train_split_"+str(i)+"_by_"+target_key_type+".json")
        elif target_key_type == "date":
            target_keys = list(vm_dict.keys())
            target_keys = sorted(target_keys)[::-1]
            train_key, valid_key, test_key = divide_list(target_keys, frac_valid=0.1, frac_test=0.25)
            vm_train, vm_valid, vm_test = split(vm_dict, target_key_type,
                    train_key=train_key, valid_key=valid_key, test_key=test_key)
            save_json(vm_valid, p+"metadata_validation_split_by_"+target_key_type+".json")
            save_json(vm_test, p+"metadata_test_split_by_"+target_key_type+".json")
            save_json(vm_train, p+"metadata_train_split_by_"+target_key_type+".json")
    print("The data split is saved in: " + p)


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
    pos = [47, 23, 19, 15]
    neg = [32, 20, 16, 12]
    gold_pos = [47]
    gold_neg = [32]
    for v in vm:
        k = to_key(v, target_key_type)
        label = v["label_state_admin"] # TODO: need to change this to label_state in the future
        if label in pos:
            count_vm[k]["pos"] += 1
        elif label in neg:
            count_vm[k]["neg"] += 1
        if label in gold_pos:
            count_vm[k]["gold_pos"] += 1
        elif label in gold_neg:
            count_vm[k]["gold_neg"] += 1
    print(json.dumps(count_vm, indent=4))


def to_key(v, target_key_type):
    # The file name contains information about camera, date, and bounding box
    # We can use this as the key of the dataset for separating training, validation, and test sets
    key = v["file_name"].split("-")
    if target_key_type == "camera":
        return "-".join(key[0:2])
    elif target_key_type == "date":
        return "-".join(key[2:5])


# Split metadata into training, validation, and test sets
def main(argv):
    # Check
    if len(argv) > 1:
        if argv[1] != "confirm":
            print("Must confirm by running: python split_metadata.py confirm")
            return
    else:
        print("Must confirm by running: python split_metadata.py confirm")
        return

    vm = load_json("../data/metadata.json")
    #method = "random"
    method = "assign"
    split_and_save_data(vm, "date", method=method)
    split_and_save_data(vm, "camera", method=method)


if __name__ == "__main__":
    main(sys.argv)
