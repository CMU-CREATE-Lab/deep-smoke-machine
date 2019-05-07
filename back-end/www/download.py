import sys
from util import *

def main(argv):
    # Read dataset
    dataset = load_json("../data/dataset.json")
    
    # Download all videos in the dataset
    video_root_path = "../data/videos/"
    check_and_create_dir(video_root_path)
    for camera in dataset:
        for date in dataset[camera]:
            for bbox in dataset[camera][date]:
                for v in dataset[camera][date][bbox]:
                    file_path = video_root_path + v["file_name"] + ".mp4"
                    if not is_file_here(file_path):
                        print("Download video", v["id"])
                        download_video(v["url_root"] + v["url_part"], file_path)
    print("Done.")

if __name__ == "__main__":
    main(sys.argv)
