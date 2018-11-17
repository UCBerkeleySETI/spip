#!/usr/bin/env python

###############################################################################
#  
#     Copyright (C) 2018 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################

import sys, traceback, math
from time import sleep

from spip_tcs import TCSServerDaemon,TCSReportingThread
from uwb_config import UWBConfig
from spip.utils import catalog


DAEMONIZE = True
DL = 1

###############################################################################
# 
class UWBTCSServerDaemon(TCSServerDaemon):

  def __init__ (self, name):
    TCSServerDaemon.__init__(self, name)
    self.valid_pulsars = []

  def range_inclusive (self, lower, upper, value):
    """ Test if value is between range of lower to upper inclusive. """

    if value < lower:
      return (False, str(value) + " was below lower limit [" + str(lower) + "]")
    if value > upper:
      return (False, str(value) + " was above upper limit [" + str(upper) + "]")
    return (True, "")

  def power_of_two (self, value):
    """ Test if value is a power of two. """

    log2_value = math.log (value,2)
    power2_value = math.pow(2, int(log2_value))
    self.log(2, "UWBTCSServerDaemon::power_of_two value=" + str(value) + \
             " log2_value=" + str(log2_value) + " power2_value=" + str(power2_value))
    if not int(value) == int(power2_value):
      return (False, str(value) + " was not a power of two")
    return (True, "")

  def power_of_two_range_inclusive (self, lower, upper, value):
    """ Test if value is a power of two and in range lower to upper. """

    (ok, message) = self.range_inclusive (lower, upper, value)
    if not ok:
      return (ok, message)
    (ok, message) = self.power_of_two (value)
    if not ok:
      return (ok, message)
    return (True, "")

  def in_list (self, valid_values, value):
    if value in valid_values:
      return (True, "")
    else:
      return (False, str(value) + " was not in the allowed list " + str(valid_values))

  def validate_config (self, config):

    self.log(2, "UWBTCSServerDaemon::validate_config()")

    source_config = config["source_parameters"]

    for istream in range(int(config["stream_configuration"]["nstream"]["#text"])):

      stream_config = config["stream" + str(istream)]
      self.log(2, "UWBTCSServerDaemon::validate_config stream=" + str(istream))

      (ok, message) = self.validate_fold_configuration(stream_config, source_config)
      if not ok:
        return (ok, message)

      (ok, message) = self.validate_search_configuration(stream_config)
      if not ok:
        return (ok, message)

      (ok, message) = self.validate_continuum_configuration(stream_config)
      if not ok:
        return (ok, message)

    return (True, "")

  def validate_continuum_configuration (self, config):

    mode_active = config["processing_modes"]["continuum"]["#text"]
    if not ((mode_active == "1") or (mode_active == "true")):
      return (True, "")

    c = config["continuum_processing_parameters"]

    prefix = "stream0, continuum, "

    (ok, message) = self.power_of_two_range_inclusive (512, 2097152, int(c["output_nchan"]["#text"]))
    if not ok:
      return (False, prefix + "output_nchan: " + message)

    (ok, message) = self.range_inclusive (0.25, 60, float(c["output_tsamp"]["#text"]))
    if not ok:
      return (False, prefix + "output_tsamp: " + message)

    (ok, message) = self.range_inclusive (10, 3600, float(c["output_tsubint"]["#text"]))
    if not ok:
      return (False, prefix + "output_tsubint: " + message)

    (ok, message) = self.in_list([1,2,3,4], int(c["output_npol"]["#text"]))
    if not ok:
      return (False, prefix + "output_npol: " + message)

    # validate the maximum data rate
    nchan = int(c["output_nchan"]["#text"])
    nbit = 32
    tsamp = float(c["output_tsamp"]["#text"])
    npol = int(c["output_npol"]["#text"])

    data_rate_bits_per_second = float (nchan * nbit * npol) / tsamp
    self.log (1, "UWBTCSServerDaemon::validate_continuum_configuration data_rate=" + str(data_rate_bits_per_second/1e9) + " Gb/s")
    data_rate_limit = 536870912
    if data_rate_bits_per_second > data_rate_limit:
      return (False, prefix + " data rate [" + str(data_rate_bits_per_second/1e9) + "] exceeds limit [" + str(data_rate_limit/1e9) + "]")


    return (True, "")

  def validate_search_configuration (self, config):

    mode_active = config["processing_modes"]["search"]["#text"]
    if not ((mode_active == "1") or (mode_active == "true")):
      return (True, "")

    c = config["search_processing_parameters"]

    prefix = "stream0, search, "

    (ok, message) = self.power_of_two_range_inclusive (8, 4096, int(c["output_nchan"]["#text"]))
    if not ok:
      return (False, prefix + "output_nchan: " + message)

    (ok, message) = self.range_inclusive (-1, 3000, float(c["custom_dm"]["#text"]))
    if not ok:
      return (False, prefix + "custom_dm: " + message)

    (ok, message) = self.in_list([1,2,4, 8, 16], int(c["output_nbit"]["#text"]))
    if not ok:
      return (False, prefix + "output_nbit: " + message)

    (ok, message) = self.power_of_two_range_inclusive (1, 1048576, int(c["output_tsamp"]["#text"]))
    if not ok:
      return (False, prefix + "output_tsamp: " + message)

    (ok, message) = self.range_inclusive (15, 3600, int(c["output_tsubint"]["#text"]))
    if not ok:
      return (False, prefix + "output_tsubint: " + message)

    (ok, message) = self.in_list([1,2,3,4], int(c["output_npol"]["#text"]))
    if not ok:
      return (False, prefix + "output_npol: " + message)

    #(ok, message) = self.in_list([0, 1, "false", "true"], int(c["co_dedisp"]["#text"]))
    #if not ok:
    #  return (False, prefix + "co_dedisp: " + message)

    # validate the maximum data rate
    nchan = int(c["output_nchan"]["#text"])
    nbit = int(c["output_nbit"]["#text"])
    tsamp_us = int(c["output_tsamp"]["#text"])
    npol = int(c["output_npol"]["#text"])
    if npol == 3:
      npol = 4

    data_rate_bits_per_second = (nchan * nbit * npol * 1000000) / tsamp_us
    self.log (1, "UWBTCSServerDaemon::validate_search_configuration data_rate=" + str(data_rate_bits_per_second/1e9) + " Gb/s")
    data_rate_limit = 1073741824
    if data_rate_bits_per_second > data_rate_limit:
      return (False, prefix + " data rate [" + str(data_rate_bits_per_second/1e9) + "] exceeds limit [" + str(data_rate_limit/1e9) + "]")

    return (True, "")

  def validate_fold_configuration (self, config, source_config):

    mode_active = config["processing_modes"]["fold"]["#text"]
    if not ((mode_active == "1") or (mode_active == "true")):
      return (True, "")

    c = config["fold_processing_parameters"]

    (ok, message) = self.power_of_two_range_inclusive (64, 4096, int(c["output_nchan"]["#text"]))
    if not ok:
      return (False, "stream0, fold, output_nchan: " + message)

    (ok, message) = self.range_inclusive (-1, 3000, float(c["custom_dm"]["#text"]))
    if not ok:
      return (False, "stream0, fold, custom_dm: " + message)

    (ok, message) = self.power_of_two_range_inclusive (8, 4096, int(c["output_nbin"]["#text"]))
    if not ok:
      return (False, "stream0, fold, output_nbin: " + message)

    (ok, message) = self.in_list([-1,1,2,3,4], int(c["output_npol"]["#text"]))
    if not ok:
      return (False, "stream0, fold, output_npol: " + message)

    (ok, message) = self.range_inclusive (8, 60, int(c["output_tsubint"]["#text"]))
    if not ok:
      return (False, "stream0, fold, output_tsubint: " + message)

    (ok, message) = self.in_list(["PSR", "CAL"], c["mode"]["#text"])
    if not ok:
      return (False, "stream0, fold, mode: " + message)

    #(ok, message) = self.range_inclusive(0.0005, int(c["output_tsubint"]["#text"])/2, int(c["custom_period"]["#text"]))
    #if not ok:
    #   return (False, "stream0, fold, custom_period: " + message)

    (ok, message) = self.in_list (["0", "1", "true", "false"], c["sk"]["#text"])
    if not ok:
      return (False, "stream0, fold, sk: " + message)

    (ok, message) = self.range_inclusive(3, 6, int(c["sk_threshold"]["#text"]))
    if not ok:
      return (False, "stream0, fold, sk_threshold: " + message)

    (ok, message) = self.power_of_two_range_inclusive(128, 4096, int(c["sk_nsamps"]["#text"]))
    if not ok:
      return (False, "stream0, fold, sk_nsamps: " + message)

    (ok, message) = self.in_list (["0", "1", "true", "false"], c["append_output"]["#text"])
    if not ok:
      return (False, "stream0, fold, append_output: " + message)

    # test validity of target name
    target = source_config["name"]["#text"]
    if c["mode"]["#text"] == "PSR":
      if not target in self.valid_pulsars:
        (ok, message) = catalog.test_pulsar_valid (target)
        if not ok:
          return (False, "target " + target + " did not exist in pulsar catalgoue")
        else:
          self.valid_pulsars.append(target)
    #else:
    #  fluxcal_on_file = self.script.cfg["CONFIG_DIR"] + "/fluxcal.on"
    #  fluxcal_off_file = self.script.cfg["CONFIG_DIR"] + "/fluxcal.off"
    #  catalog.test_fluxcal (target, fluxcal_on_file, fluxcal_off_file);

    return (True, "")

##################################################################################
# main

if __name__ == "__main__":

  if len(sys.argv) != 2:
    print "ERROR: 1 command line argument expected"
    sys.exit(1)

  # this should come from command line argument
  beam_id = sys.argv[1]

  if not int(beam_id) == -1:
    print "ERROR: only valid in server mode"
    sys.exit(1)

  script = UWBTCSServerDaemon ("uwb_tcs")

  # ensure the recv daemons can bind as they see fit
  script.cpu_list = "-1"
  state = script.configure (DAEMONIZE, DL, "tcs", "tcs")
  if state != 0:
    sys.exit(state)

  script.log(1, "STARTING SCRIPT")

  try:

    reporting_thread = TCSReportingThread(script, beam_id)
    reporting_thread.start()

    script.main (beam_id)

    reporting_thread.join()

  except:
    script.quit_event.set()
    script.log(-2, "exception caught: " + str(sys.exc_info()[0]))
    print '-'*60
    traceback.print_exc(file=sys.stdout)
    print '-'*60

  script.log(1, "STOPPING SCRIPT")
  script.conclude()
  sys.exit(0)
