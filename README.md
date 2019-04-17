# Data Collection System for Marine Engineering

Author: [@PENG Zhenghao](https://github.com/PengZhenghao)


# Get Start!

Example usage:

1. Run the data collector for nearly half hour:

```
    python recorder.py --exp-name example --timestep 180
```

or

```
    python recorder.py --exp-name example -t 180
```

2. Run the data collector without pre-defining the time duration:

```
    python recorder.py --exp-name example_infinite_time
```
 
(Note that you should press Ctrl+C to terminate this program!)

3. Run program with extra arguments:

```
    --monitoring (-m): use opencv to show the live video
    --sync (-s): use the sync version of the Recorder
    --use-h5py-video-writer (-u): write each frame of video to h5 file (This will cause extremely large file and slow down the whole program.)
    --log-level: choose from ["WARNING", "INFO", "DEBUG"], determine the verbosity.
```

4. Load the stored data:

```
# This is a python script
from recorder import Recorder

config = {"exp_name": "example", "save_dir": "examples", "use_video_writer": False}
r = Recorder(config)
data = r.read()
lidar_data = data["lidar_data"][:]
frames = data["frame"][:]
extra_data = data["extra_data"][:]

print("lidar_data contains {} and its shape is {}.".format(lidar_data, lidar_data.shape))
print("frame means for each datapoint {}.".format(frames.mean(1).mean(1).mean(1)))

r.display()
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
