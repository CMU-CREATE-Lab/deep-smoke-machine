import os
import sys
from util import *
import numpy as np
from multiprocessing import Pool

rgb_perturb_dir = "../data/rgb_perturb/"
rgb_dir = "../data/rgb/"


# Perturb rgb frames and save them
def main(argv):
    num_workers = 6
    check_and_create_dir(rgb_perturb_dir)
    file_list = get_all_file_names_in_folder(rgb_dir)
    p = Pool(num_workers)
    p.map(perturb, file_list)
    print("Done")


def perturb(f):
    print("Process: " + f)
    p = rgb_perturb_dir + f
    if not is_file_here(p):
        print("Process file %s" % f)
        frames = np.load(rgb_dir + f)
        frames_perturb = np.random.permutation(frames)
        np.save(p, frames_perturb)


if __name__ == "__main__":
    main(sys.argv)
