import os
import shutil


def mkdir_when_not_exist(abs_path):
    try:
        os.mkdir(abs_path)
    except FileExistsError:
        pass


def rmdir_when_any(abs_path):
    shutil.rmtree(abs_path)
