import sys

def add_path(p):
    if not p in sys.path :
        sys.path.append(p)

add_path("../common")
