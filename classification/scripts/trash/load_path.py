import sys
from pathlib import Path


def  sys_append_abs():
    file = Path('__file__').resolve()
    parent, root = file.parent, file.parents[1]
    sys.path.append(str(root))

