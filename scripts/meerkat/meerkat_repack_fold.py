#!/usr/bin/env python

##############################################################################
#
#     Copyright (C) 2018 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
#
###############################################################################
#
#     MeerKAT Repack Fold Beam:
#       combines output sub-bands into single archive
#       provides socket based reporting
#

import sys
import traceback

# import from spip_repack
from spip_repack import RepackBeamDaemon, RepackReportingThread

DAEMONIZE = True
DL = 1


class MeerKATRepackFoldBeamDaemon(RepackBeamDaemon):

    def __init__(self, name):
        RepackBeamDaemon.__init__(self, name)

        self.convert_psrfits = False
        self.zap_psh_script = "zap_meerkat.psh"
        self.nchan_plot = 512

###############################################################################

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print "ERROR: 1 command line argument expected"
        sys.exit(1)

    # this should come from command line argument
    beam_id = sys.argv[1]

    if int(beam_id) == -1:
        print "ERROR: MeerKATRepackFoldBeamDaemon can not operate on server"
        sys.exit(1)
    else:
        script = MeerKATRepackFoldBeamDaemon("meerkat_repack_fold", beam_id)

    state = script.configure(DAEMONIZE, DL, "repack_fold",
                             "repack_fold")
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
