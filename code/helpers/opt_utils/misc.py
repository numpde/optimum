# RA, 2021-06-11

import json
import inspect
import datetime
import pathlib

class Section:
    def __init__(self, desc, out=print):
        co = inspect.currentframe().f_back.f_code.co_name
        (self.desc, self.out) = (f"{co}: {desc}", out)

    def __enter__(self):
        self.out and self.out(f"<{self.desc}>")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # self.out and self.out(f"[Done]")
        return


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.timedelta):
            return f"{obj.total_seconds()}"
        elif isinstance(obj, pathlib.Path):
            return str(obj)
        else:
            return json.JSONEncoder.default(self, obj)
