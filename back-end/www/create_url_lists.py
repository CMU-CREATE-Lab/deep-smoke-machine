# This script creates the url lists for running the smoke recognition code
from recognize_smoke import *
from util import *

# You need to manually fill in the urls, using http://mon.createlab.org/
urls = []

# Generate the url list
url_root = "https://thumbnails-v2.createlab.org/thumbnail"
url_list = []
date_str = None
for i in range(len(urls)):
    url = urls[i]
    if url == "" or url is None: continue
    if date_str is None:
        ds = get_datetime_str_from_url(url)
        date_str = ds
    cn = get_camera_name_from_url(url)
    b = get_bound_from_url(url)
    url_list.append({
        "url": url_root + get_url_part(cam_name=cn, ds=date_str, b=b, sf=4675),
        "cam_id": cam_name_to_id(cn),
        "view_id": i})

# Save the url list
p = "../data/production_url_list/" + date_str + ".json"
save_json(url_list, p)
print("Create file at %s" % p)
