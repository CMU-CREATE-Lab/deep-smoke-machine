import sys
from collections import defaultdict
import json
import numpy as np
from util import *

# Split metadata into training, validation, and test sets
def main(argv):
    vm = load_json("../data/metadata.json")

    # Index metadata by date, camera, or bounding box
    vm_by_date = defaultdict(list)
    count_vm_by_date = defaultdict(lambda: defaultdict(int))
    count_vm_by_cam = defaultdict(lambda: defaultdict(int))
    pos = [47, 23, 19, 15]
    neg = [32, 20, 16, 12]
    gold_pos = [47]
    gold_neg = [32]
    for v in vm:
        # The file name contains information about camera, date, and bounding box
        # We can use this as the key of the dataset for separating training, validation, and test sets
        key = v["file_name"].split("-")
        cam = "-".join([key[0]] + key[4:8])
        date = "-".join(key[1:4])
        vm_by_date[date].append(v)
        label = v["label_state_admin"]
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
    
    # Print label distribution
    #print(json.dumps(count_vm_by_date, indent=4))
    #print(json.dumps(count_vm_by_cam, indent=4))

    # Randomly split dataset by date
    dates = list(vm_by_date.keys())
    np.random.shuffle(dates)
    n_dates = len(dates)
    n_valid = int(n_dates/10)
    n_test = int(n_dates/5)
    dates_valid = dates[:n_valid]
    dates_test = dates[n_valid:n_valid+n_test]
    vm_valid = []
    vm_test = []
    vm_train = []
    for d in vm_by_date:
        if d in dates_valid:
            vm_valid += vm_by_date[d]
        elif d in dates_test:
            vm_test += vm_by_date[d]
        else:
            vm_train += vm_by_date[d]
    print("Size of validation set:", len(vm_valid))
    print("Size of test set:", len(vm_test))
    print("Size of training set:", len(vm_train))

    # Save the splits
    save_json(vm_valid, "../data/metadata_validation.json")
    save_json(vm_test, "../data/metadata_test.json")
    save_json(vm_train, "../data/metadata_train.json")

if __name__ == "__main__":
    main(sys.argv)
