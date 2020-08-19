# This script creates the url lists for running the smoke recognition code
from recognize_smoke import *
from util import *

# You need to manually fill in the urls, using http://mon.createlab.org/
urls = [
    "https://thumbnails-v2.createlab.org/thumbnail?root=https://tiles.cmucreatelab.org/ecam/timemachines/clairton1/2020-01-31.timemachine/&width=720&height=720&startFrame=2051&format=png&fps=12&tileFormat=mp4&startDwell=0&endDwell=0&boundsLTRB=6251.679973016499,1434.5614772593433,6803.976846311334,1986.8583505541783&labelsFromDataset&nframes=1",
    "https://thumbnails-v2.createlab.org/thumbnail?root=https://tiles.cmucreatelab.org/ecam/timemachines/clairton1/2020-01-31.timemachine/&width=720&height=720&startFrame=2051&format=png&fps=12&tileFormat=mp4&startDwell=0&endDwell=0&boundsLTRB=5913.034784917295,1440.3751285571848,6465.33165821213,1992.6720018520195&labelsFromDataset&nframes=1",
    "https://thumbnails-v2.createlab.org/thumbnail?root=https://tiles.cmucreatelab.org/ecam/timemachines/clairton1/2020-01-31.timemachine/&width=720&height=720&startFrame=2051&format=png&fps=12&tileFormat=mp4&startDwell=0&endDwell=0&boundsLTRB=5501.001418926446,1473.881239234236,6011.442535575285,1984.3223558830741&labelsFromDataset&nframes=1",
    "https://thumbnails-v2.createlab.org/thumbnail?root=https://tiles.cmucreatelab.org/ecam/timemachines/clairton1/2020-01-31.timemachine/&width=720&height=720&startFrame=2051&format=png&fps=12&tileFormat=mp4&startDwell=0&endDwell=0&boundsLTRB=5178.343364992632,1459.2775838793675,5710.0440884399,1990.9783073266367&labelsFromDataset&nframes=1",
    "https://thumbnails-v2.createlab.org/thumbnail?root=https://tiles.cmucreatelab.org/ecam/timemachines/clairton1/2020-01-31.timemachine/&width=720&height=720&startFrame=2051&format=png&fps=12&tileFormat=mp4&startDwell=0&endDwell=0&boundsLTRB=4738.3008699899065,1458.5508075629114,5263.343570835373,1983.5935084083785&labelsFromDataset&nframes=1",
    "https://thumbnails-v2.createlab.org/thumbnail?root=https://tiles.cmucreatelab.org/ecam/timemachines/clairton1/2020-01-31.timemachine/&width=720&height=720&startFrame=2051&format=png&fps=12&tileFormat=mp4&startDwell=0&endDwell=0&boundsLTRB=4213.258169144423,1494.4747818312862,4738.30086998989,2019.5174826767534&labelsFromDataset&nframes=1",
    "https://thumbnails-v2.createlab.org/thumbnail?root=https://tiles.cmucreatelab.org/ecam/timemachines/clairton1/2020-01-31.timemachine/&width=720&height=720&startFrame=2051&format=png&fps=12&tileFormat=mp4&startDwell=0&endDwell=0&boundsLTRB=3877.523612285187,1521.3867761612012,4383.9986607541105,2027.8618246301248&labelsFromDataset&nframes=1"
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
