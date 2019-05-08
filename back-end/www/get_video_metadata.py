import sys
from collections import defaultdict
import requests
from util import *

def main(argv):
    print("Get video metadata...")
    get_video_metadata("../data/video_metadata.json")
    print("Done.")

# Get video metadata from the video-labeling-tool and save to a file
def get_video_metadata(file_path=None):
    # Read the user token
    # The token is obtained from https://smoke.createlab.org/gallery.html in dashboard mode
    user_token = load_json("../data/user_token.json")["user_token"]

    # Request dataset
    url_root = "https://api.smoke.createlab.org/api/v1/"
    videos = iterative_query(url_root+"get_pos_gold_labels", user_token)
    videos += iterative_query(url_root+"get_neg_gold_labels", user_token)
    videos += iterative_query(url_root+"get_pos_labels_by_researcher", user_token)
    videos += iterative_query(url_root+"get_neg_labels_by_researcher", user_token)

    # Build dataset
    dataset = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for v in videos:
        # The file name contains information about camera, date, and bounding box
        # We can use this as the key of the dataset for separating training, validation, and testing sets
        key = v["file_name"].split("-")
        camera = key[0]
        date = "-".join(key[1:4])
        bbox = "-".join(key[4:8])
        dataset[camera][date][bbox].append(v)

    # Save and return dataset
    if file_path is not None:
        check_and_create_dir(file_path)
        save_json(dataset, file_path)
    return dataset

# Query a paginated api call iteratively until getting all the data
def iterative_query(url, user_token, page_size=1000, page_number=1):
    r = requests.post(url=url, data={"user_token": user_token, "pageSize": page_size, "pageNumber": page_number})
    r = r.json()
    total = r["total"]
    r = r["data"]
    if page_size*page_number >= total:
        return r
    else:
        return r + iterative_query(url, user_token, page_size=page_size, page_number=page_number+1)

if __name__ == "__main__":
    main(sys.argv)
