# src/utils/logger.py
import csv, os, time
from collections import defaultdict

class CSVLogger:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._fh = open(path, "w", newline="", encoding="utf-8")
        self._w = None

    def write(self, row: dict):
        if self._w is None:
            self._w = csv.DictWriter(self._fh, fieldnames=list(row.keys()))
            self._w.writeheader()
        self._w.writerow(row)
        self._fh.flush()

    def close(self):
        try: self._fh.close()
        except: pass
