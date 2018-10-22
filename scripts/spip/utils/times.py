###############################################################################
# 
#     Copyright (C) 2015 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
#
###############################################################################

from datetime import datetime
from datetime import timedelta
from time import mktime
from calendar import timegm

def getCurrentTimeUS():
  now = datetime.today()
  now_str = now.strftime("%Y-%m-%d-%H:%M:%S.%f")
  return now_str

def getCurrentTimeMS():
  return getCurrentTimeUS()[:-3]

def getCurrentTime(toadd=0):
  now = datetime.today()
  if (toadd > 0):
    delta = timedelta(0, toadd)
    now = now + delta
  now_str = now.strftime("%Y-%m-%d-%H:%M:%S")
  return now_str

def getUTCTime(toadd=0):
  now = datetime.utcnow()
  if (toadd > 0):
    delta = timedelta(0, toadd)
    now = now + delta
  now_str = now.strftime("%Y-%m-%d-%H:%M:%S")
  return now_str

def convertLocalToUnixTime(epoch_str):
  epoch = datetime.strptime(epoch_str, "%Y-%m-%d-%H:%M:%S")
  return mktime(epoch.timetuple())

def convertUTCToUnixTime(epoch_str):
  epoch = datetime.strptime(epoch_str, "%Y-%m-%d-%H:%M:%S")
  return timegm(epoch.timetuple())

def diffUTCTimes(epoch1_str, epoch2_str):
    epoch1 = datetime.strptime(epoch1_str, "%Y-%m-%d-%H:%M:%S")
    epoch2 = datetime.strptime(epoch2_str, "%Y-%m-%d-%H:%M:%S")
    delta = epoch2 - epoch1
    return delta.seconds

def diffUTCTime(epoch_str):
    epoch = datetime.strptime(epoch_str, "%Y-%m-%d-%H:%M:%S")
    now = datetime.utcnow()
    delta = now - epoch
    return delta.seconds

def getCurrentDateTime():
  return datetime.today()

def diffCurrentDateTime(epoch):
  now = datetime.today()
  delta = now - epoch
  return delta.seconds

def getCurrentTimeFromUnix (unixtime):
  epoch = datetime.fromtimestamp(unixtime)
  epoch_str = epoch.strftime("%Y-%m-%d-%H:%M:%S")
  return epoch_str

