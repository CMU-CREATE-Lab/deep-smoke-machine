import sys
from collections import defaultdict
import json
import numpy as np
from util import *

def random_split(vm_dict):
    target_keys = list(vm_dict.keys())
    np.random.shuffle(target_keys)
    n_keys = len(target_keys)
    n_valid = int(n_keys*0.1) # 10% for validation
    n_test = int(n_keys*0.3) # 30% for testing
    target_keys_valid = target_keys[:n_valid]
    target_keys_test = target_keys[n_valid:n_valid+n_test]
    vm_valid = []
    vm_test = []
    vm_train = []
    for d in vm_dict:
        if d in target_keys_valid:
            vm_valid += vm_dict[d]
        elif d in target_keys_test:
            vm_test += vm_dict[d]
        else:
            vm_train += vm_dict[d]
    print("Size of validation set:", len(vm_valid))
    print("Size of test set:", len(vm_test))
    print("Size of training set:", len(vm_train))
    return (vm_train, vm_valid, vm_test)


def print_distribution(vm):
    count_vm_by_date = defaultdict(lambda: defaultdict(int))
    count_vm_by_cam = defaultdict(lambda: defaultdict(int))
    pos = [47, 23, 19, 15]
    neg = [32, 20, 16, 12]
    gold_pos = [47]
    gold_neg = [32]
    for v in vm:
        cam, date = to_keys(v)
        label = v["label_state_admin"] # TODO: need to change this to label_state in the future
        # By date
        if label in pos:
            count_vm_by_date[date]["pos"] += 1
        elif label in neg:
            count_vm_by_date[date]["neg"] += 1
        if label in gold_pos:
            count_vm_by_date[date]["gold_pos"] += 1
        elif label in gold_neg:
            count_vm_by_date[date]["gold_neg"] += 1
        # By camera
        if label in pos:
            count_vm_by_cam[cam]["pos"] += 1
        elif label in neg:
            count_vm_by_cam[cam]["neg"] += 1
        if label in gold_pos:
            count_vm_by_cam[cam]["gold_pos"] += 1
        elif label in gold_neg:
            count_vm_by_cam[cam]["gold_neg"] += 1
    #print(json.dumps(count_vm_by_date, indent=4))
    print(json.dumps(count_vm_by_cam, indent=4))


def to_keys(v):
    # The file name contains information about camera, date, and bounding box
    # We can use this as the key of the dataset for separating training, validation, and test sets
    key = v["file_name"].split("-")
    cam = "-".join(key[0:2])
    date = "-".join(key[2:5])
    return (cam, date)


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

    # Index metadata by date, camera, or bounding box
    vm = load_json("../data/metadata.json")
    vm_by_date = defaultdict(list)
    vm_by_cam = defaultdict(list)
    for v in vm:
        cam, date = to_keys(v)
        vm_by_date[date].append(v)
        vm_by_cam[cam].append(v)

    # Randomly split dataset
    #vm_train, vm_valid, vm_test = random_split(vm_by_date)
    vm_train, vm_valid, vm_test = random_split(vm_by_cam)
    print("\nTraining:")
    print_distribution(vm_train)
    print("\nValidation:")
    print_distribution(vm_valid)
    print("\nTesting:")
    print_distribution(vm_test)

    # Save the splits
    save_json(vm_valid, "../data/metadata_validation.json")
    save_json(vm_test, "../data/metadata_test.json")
    save_json(vm_train, "../data/metadata_train.json")


if __name__ == "__main__":
    main(sys.argv)
