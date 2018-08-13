#!/usr/bin/env python

##############################################################################
#
#     Copyright (C) 2018 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
#
###############################################################################

import sys
import traceback

from spip_stat import StatDaemon, StatReportingThread

DAEMONIZE = True
DL = 1


class UWBStatDaemon(StatDaemon):

    def __init__(self, name, id):
        StatDaemon.__init__(self, name, str(id))

        self.gen_freqtime = False
        self.gen_timeseries = True
        self.gen_bandpass = True
        self.gen_histogram = True


###############################################################################

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print "ERROR: 1 command line argument expected"
        sys.exit(1)

    # this should come from command line argument
    stream_id = sys.argv[1]

    script = UWBStatDaemon("spip_stat", stream_id)

    state = script.configure(DAEMONIZE, DL, "stat", "stat")

    if state != 0:
        sys.exit(state)

    script.log(1, "STARTING SCRIPT")

    try:

        reporting_thread = StatReportingThread(script, stream_id)
        reporting_thread.start()
        script.main()
        reporting_thread.join()

    except Exception as e:

        script.log(-2, "exception caught: " + str(sys.exc_info()[0]))
        formatted_lines = traceback.format_exc().splitlines()
        script.log(0, '-'*60)
        for line in formatted_lines:
            script.log(0, line)
        script.log(0, '-'*60)

        print '-'*60
        traceback.print_exc(file=sys.stdout)
        print '-'*60
        script.quit_event.set()

    script.log(1, "STOPPING SCRIPT")
    script.conclude()
    sys.exit(0)
