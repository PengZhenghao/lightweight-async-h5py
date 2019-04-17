class RecorderConfig(object):
    metadata = {
    "buffer_size": 5,
    "save_dir": "experiment",
    "compress": "gzip",
    "dataset_names": ("lidar_data", "extra_data", "frame", "timestamp"),
    "dataset_dtypes": {"lidar_data": "uint16", "extra_data": "float32", "frame": "uint8", "timestamp": "float64"},
    "dataset_shapes": {"lidar_data": (30600,), "extra_data": (8,), "frame": (960, 1280, 3), "timestamp": (1,)},
    "use_video_writer": True,
        "frame_rate": 10,
        "log_interval": 10
    }


