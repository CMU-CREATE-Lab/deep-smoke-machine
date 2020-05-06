import sys
import urllib.request
from util import *


# Download all videos in the metadata json file
def main(argv):
    vm = load_json("../data/metadata_all.json")["data"]
    video_root_path = "../data/videos_all/"
    check_and_create_dir(video_root_path)
    problem_video_ids = []
    for v in vm:
        file_path = video_root_path + v["file_name"] + ".mp4"
        if is_file_here(file_path): continue # skip if file exists
        print("Download video", v["id"])
        try:
            url = v["url_root"] + v["url_part"]
            url = url.replace("width=180", "width=320")
            url = url.replace("height=180", "height=320")
            urllib.request.urlretrieve(url, file_path)
        except:
            print("\tError downloading video", v["id"])
            problem_video_ids.append(v["id"])
    print("Done download_all_videos.py")
    if len(problem_video_ids) > 0:
        print("The following videos were not downloaded due to errors:")
        for i in problem_video_ids:
            print("\ti\n")


if __name__ == "__main__":
    main(sys.argv)
