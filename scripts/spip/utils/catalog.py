###############################################################################
#  
#     Copyright (C) 2016 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################

from spip.utils.core import system

# test whether the specified target exists in the pulsar catalog
def test_pulsar_valid (target):

  (reply, message) = get_psrcat_param (target, "jname")
  if not reply:
    return (reply, message)

  if message == target:
    return (True, "")
  else:
    return (False, "pulsar " + target + " did not exist in catalog")

def get_psrcat_param (target, param):

  # remove the _R suffix
  if target.endswith('_R'):
    target = target[:-2]

  cmd = "psrcat -all " + target + " -c " + param + " -nohead -o short"
  rval, lines = system (cmd)
  if rval != 0 or len(lines) <= 0:
    return (False, "could not use psrcat")

  if lines[0].startswith("WARNING"):
    return (False, "pulsar " + target + " did not exist in catalog")

  parts = lines[0].split()
  if len(parts) == 2 and parts[0] == "1":
    return (True, parts[1])
  else:
    return (False, "pulsar " + target + " did not match only 1 entry")

def test_fluxcal (target, fluxcal_on_file, fluxcal_off_file):

  # check if the target matches the fluxcal off file
  cmd = "grep " + target + " " + fluxcal_on_file + " | wc -l"
  rval, lines = system (cmd)
  if rval == 0 and len(lines) == 1 and int(lines[0]) > 0:
    return (True, "")

  # check if the target matches the fluxcal off file
  cmd = "grep " + target + " " + fluxcal_on_file + " | wc -l"
  rval, lines = system (cmd)
  if rval == 0 and len(lines) == 1 and int(lines[0]) > 0:
    return (True, "")

  return (False, target + " did not exist in fluxcal files")
