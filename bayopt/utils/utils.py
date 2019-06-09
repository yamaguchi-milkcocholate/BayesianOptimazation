import os


def mkdir_when_not_exist(abs_path):
    try:
        os.mkdir(abs_path)
    except FileExistsError:
        pass
