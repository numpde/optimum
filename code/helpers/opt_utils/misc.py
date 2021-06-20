# RA, 2021-06-11
# RA, 2021-06-20

import json
import datetime
import pathlib
import time
import inspect
import sorcery
import executing

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


class Memo(dict):
    """
    Example:
        memo = Memo()
        (name, surname) = memo("X", "Y")
        print(memo)  # {'name': 'X', 'surname': 'Y'}

    Modified from `sorcery.spell` and `sorcery.assigned_names`.
    """

    def __call__(self, value0, *values):
        exe = executing.Source.executing(inspect.currentframe().f_back)
        names = sorcery.core.FrameInfo(exe).assigned_names(allow_one=True)[0]

        assert len(names) == (1 + len(values))

        for (name, value) in zip(names, (value0, *values)):
            self[name] = value

        if values:
            return (value0, *values)
        else:
            return value0

