#!/usr/bin/env python

###############################################################################
#
#     Copyright (C) 2018 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
#
###############################################################################

import sys
import traceback

from spip_tcs import TCSServerDaemon, TCSReportingThread


DAEMONIZE = True
DL = 1


class MeerKATTCSServerDaemon(TCSServerDaemon):

    def __init__(self, name):
        TCSServerDaemon.__init__(self, name)

    def validate_config(self, config):

        self.log(2, "MeerKATTCSServerDaemon::validate_config()")

        return (True, "")


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print "ERROR: 1 command line argument expected"
        sys.exit(1)

    # this should come from command line argument
    beam_id = sys.argv[1]

    if not int(beam_id) == -1:
        print "ERROR: only valid in server mode"
        sys.exit(1)

    script = MeerKATTCSServerDaemon("tcs")

    # ensure the recv daemons can bind as they see fit
    script.cpu_list = "-1"
    state = script.configure(DAEMONIZE, DL, "tcs", "tcs")
    if state != 0:
        sys.exit(state)

    script.log(1, "STARTING SCRIPT")

    try:

        reporting_thread = TCSReportingThread(script, beam_id)
        reporting_thread.start()

        script.main(beam_id)

        reporting_thread.join()

    except Exception as e:
        script.quit_event.set()
        script.log(-2, "exception caught: " + str(sys.exc_info()[0]))
        print '-'*60
        traceback.print_exc(file=sys.stdout)
        print '-'*60

    script.log(1, "STOPPING SCRIPT")
    script.conclude()
    sys.exit(0)
