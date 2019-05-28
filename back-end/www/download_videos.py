import sys
import urllib.request
from util import *

# Download all videos in the metadata json file
def main(argv):
    # Check
    if len(argv) > 1:
        if argv[1] != "confirm":
            print("Must confirm by running: python download_videos.py confirm")
            return
    else:
        print("Must confirm by running: python download_videos.py confirm")
        return

    # Download
    vm = load_json("../data/metadata.json")
    video_root_path = "../data/videos/"
    check_and_create_dir(video_root_path)
    for v in vm:
        file_path = video_root_path + v["file_name"] + ".mp4"
        if is_file_here(file_path): continue # skip if file exists
        print("Download video", v["id"])
        urllib.request.urlretrieve(v["url_root"] + v["url_part"], file_path)
    print("Done")

if __name__ == "__main__":
    main(sys.argv)
