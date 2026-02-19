# SPDX-License-Identifier: MIT

import os

def readable_dir(prospective_dir):
  if not os.path.isdir(prospective_dir):
    raise Exception("readable_dir:{0} is not a valid path".format(prospective_dir))
  if os.access(prospective_dir, os.R_OK):
    return prospective_dir
  else:
    raise Exception("readable_dir:{0} is not a readable dir".format(prospective_dir))


def writeable_dir(prospective_dir):
  if not os.path.isdir(prospective_dir):
    os.mkdir(prospective_dir)
  if os.access(prospective_dir, os.R_OK):
    return prospective_dir
  else:
    raise Exception("writeable_dir:{0} does not exist and could not be created".format(prospective_dir))
