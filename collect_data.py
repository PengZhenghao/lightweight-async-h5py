from recorder import build_recorder_process
from utils import setup_logger
import logging
import argparse
import time
import multiprocessing
import numpy as np

"""
Example Usages: 
1. Run the data collector for nearly half hour:
    
    python collect_data.py --exp-name 0101-Trimaran-Tracking --timestep 18000

or

    python collect_data.py --exp-name 0101-Trimaran-Tracking -t 18000
    
2. Run the data collector for not pre-defined time duration:

    python collect_data.py --exp-name 0101-Trimaran-Tracking
    
(Note that you should press Ctrl+C to terminate this program!)
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", default=None, type=str)
    parser.add_argument("--log-level", default="INFO", type=str)
    parser.add_argument("--timestep", "-t", default=-1, type=int)
    parser.add_argument("--fake-lidar", '-f', action="store_true", default=False)
    parser.add_argument("--monitoring", '-m', action="store_true", default=False)
    args = parser.parse_args()

    setup_logger(args.log_level)

    log_queue = multiprocessing.Queue()
    data_queue = multiprocessing.Queue()
    recorder_process = multiprocessing.Process(target=build_recorder_process, args=(
    {"exp_name": args.exp_name}, data_queue, log_queue, args.log_level, args.monitoring))
    recorder_process.start()
    now = time.time()

    st = now
    cnt = 0
    log_interval = 10

    try:
        logging.info("Start Record Data!")
        while True:
            logging.debug("The {} iteration!".format(cnt))

            data_dict = {}
            data_dict["frame"] = np.random.randint(low=0, high=256, size=(960, 1280, 3), dtype=np.uint8)
            data_dict["lidar_data"] = np.random.randint(low=0, high=30000, size=(30600,), dtype=np.uint16)
            data_dict["extra_data"] = np.random.randint(0, 1000, size=(8,), dtype=np.float64)

            data_queue.put(data_dict)

            if cnt % log_interval == 0:
                if args.timestep == -1:
                    logging.info("Data processed in frequency {}. Press Ctrl+C to terminate this program!".format(
                        log_interval / (time.time() - now)))
                else:
                    logging.info("Data processed in frequency {}.".format(log_interval / (time.time() - now)))
                now = time.time()
            cnt += 1
            if args.timestep > 0 and cnt == args.timestep:
                break
    finally:
        et = time.time()
        logging.info("Recording Finish! It take {} seconds and collect {} data! Average FPS {}.".format(et - st, cnt,
                                                                                                        cnt / (
                                                                                                        et - st)))
        data_queue.put(None)
        recorder_process.join()
