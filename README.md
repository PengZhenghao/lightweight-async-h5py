# Data Collection System for Marine Engineering

Author: [@PENG Zhenghao](https://github.com/PengZhenghao)


# Get Start!

Example usage:

1. Run the data collector for nearly half hour:

```
    python collect_data.py --exp-name 0101-Trimaran-Tracking --timestep 18000
```

or

```
    python collect_data.py --exp-name 0101-Trimaran-Tracking -t 18000
```
 
2. Run the data collector for not pre-defined time duration:

```
    python collect_data.py --exp-name 0101-Trimaran-Tracking
```
 
(Note that you should press Ctrl+C to terminate this program!)

3. Load the stored data:

```
# This is a python script
from recorder import Recorder

config = {"exp_name": "example", "save_dir": "examples", "use_video_writer": False}
r = Recorder(config)
data = r.read()
lidar_data = data["lidar_data"]
frames = data["frame"]
extra_data = data["extra_data"]

print("lidar_data contains {} and its shape is {}.".format(lidar_data, lidar_data.shape))

import cv2

for i in frames:
    cv2.imshow("example", i)
    cv2.waitKey(50)
cv2.destroyAllWindows()
```


# Data Structure

If you don't set `use_video_writer=True` in config, the data structure looks like: 

```
experiment
  +exp-name.h5  # The following is the structure of .h5 file.
    +/config (attrubite)
    +/lidar_data (dataset, a np.ndarray with shape (-1, 30600), is the raw data collected from LiDar)
    +/extra_data (dataset, a np.ndarray with shape (-1, 8), the GPS information)
    +/timestamp (dataset, a np.ndarray with shape (-1,))
    +/frame (dataset, a np.ndarray with shape (-1, 960, 1280, 3), camera captured image at the same frequence of LiDAR)
```

Other wise it looks like:

```
experiment
  +exp-name.avi
  +exp-name.h5  # The following is the structure of .h5 file.
    +/config (attrubite)
    +/lidar_data (dataset, a np.ndarray with shape (-1, 30600), is the raw data collected from LiDar)
    +/extra_data (dataset, a np.ndarray with shape (-1, 8), the GPS information)
    +/timestamp (dataset, a np.ndarray with shape (-1,))
```

Note that in current implementation, all data are synced sampled, which means they have same length in temporal dimension. 
