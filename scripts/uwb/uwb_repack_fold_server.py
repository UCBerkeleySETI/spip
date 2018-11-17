#!/usr/bin/env python

##############################################################################
#
#     Copyright (C) 2018 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
#
###############################################################################
#
#     UWB Repack Fold Server:
#       combines output sub-bands into single archive
#       provides socket based reporting
#

import sys
import traceback

# import from spip_repack
from spip_repack import RepackServerDaemon, RepackReportingThread

DAEMONIZE = True
DL = 1


class UWBRepackFoldServerDaemon(RepackServerDaemon):

    def __init__(self, name):
        RepackServerDaemon.__init__(self, name)

        self.convert_psrfits = False
        self.zap_psh_script = "zap_uwl.psh"
        self.nchan_plot = 2048

###############################################################################

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print "ERROR: 1 command line argument expected"
        sys.exit(1)

    # this should come from command line argument
    beam_id = sys.argv[1]

    if not int(beam_id) == -1:
        print "ERROR: UWBRepackFoldServerDaemon can only operate on server"
        sys.exit(1)
    else:
        script = UWBRepackFoldServerDaemon("uwb_repack_fold_server")

    state = script.configure(DAEMONIZE, DL, "uwb_repack_fold",
                             "uwb_repack_fold")
    if state != 0:
        script.quit_event.set()
        sys.exit(state)

    script.log(1, "STARTING SCRIPT")

    try:
        reporting_thread = RepackReportingThread(script, beam_id)
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
