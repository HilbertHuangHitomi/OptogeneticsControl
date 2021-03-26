# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 13:17:40 2020

@author: Hilbert Huang Hitomi
"""


import os
import win32file
import win32con
from PARAMS import hyperparameters


# give the watch path
def WatchPath():
    FILE_LIST_DIRECTORY = win32con.GENERIC_READ | win32con.GENERIC_WRITE
    path_to_watch = os.path.join(hyperparameters['path'], 'spike2data', 'record')
    hDir = win32file.CreateFile (
      path_to_watch,
      FILE_LIST_DIRECTORY,
      win32con.FILE_SHARE_READ | win32con.FILE_SHARE_WRITE,
      None,
      win32con.OPEN_EXISTING,
      win32con.FILE_FLAG_BACKUP_SEMANTICS,
      None)
    return hDir


# watch attribute changing
def Watching(hDir):
    win32file.ReadDirectoryChangesW(
        hDir,
        512,
        True,
        win32con.FILE_NOTIFY_CHANGE_ATTRIBUTES |
        win32con.FILE_NOTIFY_CHANGE_SIZE |
        win32con.FILE_NOTIFY_CHANGE_LAST_WRITE,
        None,
        None)