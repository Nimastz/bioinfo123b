# src/utils/logger.py
import csv, os

class CSVLogger:
    def __init__(self, csv_path: str, fieldnames: list[str] | None = None):
        self.csv_path = csv_path
        self.fieldnames = fieldnames  # may be None until first log
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        self._file = open(self.csv_path, "a", newline="", encoding="utf-8")
        self._writer = None  # create after we know fieldnames

    def _ensure_writer(self, row: dict):
        if self._writer is None:
            if self.fieldnames is None:
                self.fieldnames = list(row.keys())
            self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
            if self._file.tell() == 0:
                self._writer.writeheader()
                self._file.flush()

    def log(self, row: dict):
        self._ensure_writer(row)
        safe = {k: row.get(k, "") for k in self.fieldnames}
        self._writer.writerow(safe)
        self._file.flush()

    def close(self):
        try:
            self._file.flush()
            self._file.close()
        except Exception:
            pass