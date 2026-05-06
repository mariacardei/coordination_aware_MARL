import os
import csv
import time


class CSVLogger:
    def __init__(self, log_dir, filename, header):
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, filename)
        self.start_time = time.time()

        # write header once
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    def log(self, row):
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
