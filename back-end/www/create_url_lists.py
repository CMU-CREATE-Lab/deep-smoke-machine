# This script creates the url lists for running the smoke recognition code
from recognize_smoke import *
from util import *

# You need to manually fill in the urls, using http://mon.createlab.org/
urls = [
    "https://thumbnails-v2.createlab.org/thumbnail?root=https://tiles.cmucreatelab.org/ecam/timemachines/clairton1/2020-07-14.timemachine/&width=720&height=720&startFrame=9579&format=png&fps=12&tileFormat=mp4&startDwell=0&endDwell=0&boundsLTRB=6421.447232743976,1268.9111752080744,6944.055570952111,1791.519513416209&labelsFromDataset&nframes=1",
    "https://thumbnails-v2.createlab.org/thumbnail?root=https://tiles.cmucreatelab.org/ecam/timemachines/clairton1/2020-07-14.timemachine/&width=720&height=720&startFrame=9579&format=png&fps=12&tileFormat=mp4&startDwell=0&endDwell=0&boundsLTRB=6055.621395998291,1285.4145964146483,6578.229734206426,1808.0229346227827&labelsFromDataset&nframes=1",
    "https://thumbnails-v2.createlab.org/thumbnail?root=https://tiles.cmucreatelab.org/ecam/timemachines/clairton1/2020-07-14.timemachine/&width=720&height=720&startFrame=9579&format=png&fps=12&tileFormat=mp4&startDwell=0&endDwell=0&boundsLTRB=5645.786436035063,1264.7853199064325,6168.394774243198,1787.3936581145672&labelsFromDataset&nframes=1",
    "https://thumbnails-v2.createlab.org/thumbnail?root=https://tiles.cmucreatelab.org/ecam/timemachines/clairton1/2020-07-14.timemachine/&width=720&height=720&startFrame=9579&format=png&fps=12&tileFormat=mp4&startDwell=0&endDwell=0&boundsLTRB=5319.843867205247,1297.7921623195775,5842.452205413382,1820.400500527712&labelsFromDataset&nframes=1",
    "https://thumbnails-v2.createlab.org/thumbnail?root=https://tiles.cmucreatelab.org/ecam/timemachines/clairton1/2020-07-14.timemachine/&width=720&height=720&startFrame=9579&format=png&fps=12&tileFormat=mp4&startDwell=0&endDwell=0&boundsLTRB=4875.626779728322,1278.5381709119097,5398.235117936457,1801.1465091200441&labelsFromDataset&nframes=1",
    "https://thumbnails-v2.createlab.org/thumbnail?root=https://tiles.cmucreatelab.org/ecam/timemachines/clairton1/2020-07-14.timemachine/&width=720&height=720&startFrame=9579&format=png&fps=12&tileFormat=mp4&startDwell=0&endDwell=0&boundsLTRB=4343.339931226132,1314.8205102889794,4856.316372650307,1827.796951713155&labelsFromDataset&nframes=1",
    "https://thumbnails-v2.createlab.org/thumbnail?root=https://tiles.cmucreatelab.org/ecam/timemachines/clairton1/2020-07-14.timemachine/&width=720&height=720&startFrame=9579&format=png&fps=12&tileFormat=mp4&startDwell=0&endDwell=0&boundsLTRB=4009.5822024500085,1359.1538992751698,4514.31441369238,1863.8861105175415&labelsFromDataset&nframes=1"
]

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
