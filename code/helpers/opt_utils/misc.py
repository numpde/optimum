# RA, 2021-06-11

import os
import json
import inspect
import datetime
import pathlib
import time

REPO = next(p for p in pathlib.Path(__file__).parents for p in p.glob("code")).parent.resolve()

class Section:
    def __init__(self, desc, out=print):
        co = inspect.currentframe().f_back.f_code.co_name
        (self.desc, self.out) = (f"{co}: {desc}", out)

    def __enter__(self):
        self.out and self.out(f"<{self.desc}>")
        time.sleep(1)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # self.out and self.out(f"[Done]")
        return


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.timedelta):
            return repr(obj)
        elif isinstance(obj, pathlib.Path):
            return str(obj.resolve().relative_to(REPO))
        else:
            return json.JSONEncoder.default(self, obj)
