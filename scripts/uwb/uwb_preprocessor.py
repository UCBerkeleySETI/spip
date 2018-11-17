#!/usr/bin/env python

##############################################################################
#
#         Copyright (C) 2018 by Andrew Jameson
#         Licensed under the Academic Free License version 2.1
#
###############################################################################

import os
import sys
import traceback
import threading

from spip.daemons.bases import StreamBased
from spip.daemons.daemon import Daemon
from spip.log_socket import LogSocket
from spip.utils.core import system_piped
from spip.config import Config

from spip_smrb import SMRBDaemon

DAEMONIZE = True
DL = 1


class preprocThread (threading.Thread):

    def __init__(self, cmd, dir, pipe, dl):
        threading.Thread.__init__(self)
        self.cmd = cmd
        self.pipe = pipe
        self.dir = dir
        self.dl = dl

    def run(self):
        cmd = self.cmd
        rval = system_piped(cmd, self.pipe, self.dl <= DL, work_dir=self.dir)
        return rval


class PreprocessorDaemon(Daemon, StreamBased):

    def __init__(self, name, id):
        Daemon.__init__(self, name, str(id))
        StreamBased.__init__(self, id, self.cfg)

        self.valid_plots = []
        self.results = {}

        self.results["lock"] = threading.Lock()
        self.results["valid"] = False

        (host, beam_id, sub_id) = self.cfg["STREAM_" + id].split(":")
        self.beam_name = self.cfg["BEAM_" + beam_id]

        (cfreq, bw, nchan) = self.cfg["SUBBAND_CONFIG_" + sub_id].split(":")
        self.cfreq = cfreq

        self.proc_dir = self.cfg["CLIENT_PREPROC_DIR"] + "/processing/" + \
            self.beam_name

    #################################################################
    # main
    #             id >= 0     process folded archives from a stream
    #             id == -1    process folded archives from all streams
    def main(self):

        if not os.path.exists(self.proc_dir):
            os.makedirs(self.proc_dir, 0755)

        # get the data block keys
        db_prefix = self.cfg["DATA_BLOCK_PREFIX"]
        num_stream = self.cfg["NUM_STREAM"]
        stream_id = str(self.id)
        self.debug("stream_id=" + str(self.id))

        # 4 data blocks
        in_id = self.cfg["RECEIVING_DATA_BLOCK"]
        trans_id = self.cfg["TRANSIENTS_DATA_BLOCK"]
        out_id = self.cfg["PROCESSING_DATA_BLOCK"]

        # 4 data block keys
        in_key = SMRBDaemon.getDBKey(db_prefix, stream_id, num_stream, in_id)
        trans_key = SMRBDaemon.getDBKey(db_prefix, stream_id, num_stream,
                                        trans_id)
        out_key = SMRBDaemon.getDBKey(db_prefix, stream_id, num_stream, out_id)

        log_host = self.cfg["SERVER_HOST"]
        log_port = int(self.cfg["SERVER_LOG_PORT"])

        self.debug("SMRBDaemon.waitForSMRB()")
        smrb_exists = SMRBDaemon.waitForSMRB(in_key, self)

        if not smrb_exists:
            self.error("smrb["+str(self.id)+"] no valid SMRB with " +
                       "key=" + self.db_key)
            self.quit_event.set()
            return

        # determine the number of channels to be processed by this stream
        (cfreq, bw, nchan) = self.cfg["SUBBAND_CONFIG_" + stream_id].split(":")

        # this stat command will not change from observation to observation
        preproc_cmd = "uwb_preprocessing_pipeline " + in_key + " " + \
            trans_key + " " + out_key + " -d " + \
            self.cfg["GPU_ID_" + stream_id]

        tag = "preproc" + stream_id

        # enter the main loop
        while (not self.quit_event.isSet()):

            # wait for the header to acquire the processing parameters
            cmd = "dada_header -k " + in_key + " -t " + tag
            self.debug(cmd)
            self.binary_list.append(cmd)
            rval, lines = self.system(cmd, 2, True)
            self.binary_list.remove(cmd)

            if rval != 0 or self.quit_event.isSet():
                return

            self.debug("parsing header")
            header = Config.parseHeader(lines)

            cmd = preproc_cmd

            utc_start = header["UTC_START"]
            source = header["SOURCE"]
            freq = header["FREQ"]

            # directory in which to run preprocessor
            proc_dir = self.proc_dir + "/" + utc_start + "/" + source + "/" + \
                freq

            if not os.path.exists(proc_dir):
                os.makedirs(proc_dir, 0755)

            # write the header to the proc_dir
            header_file = proc_dir + "/obs.header"
            self.debug("writing obs.header to out_dir")
            Config.writeDictToCFGFile(header, header_file)

            run_adaptive_filter = (header["ADAPTIVE_FILTER"] == "1")

            # presense of RFI reference is based on NPOL == 3
            have_rfi_reference_pol = (int(header["NPOL"]) == 3)

            # presence of a calibration signal
            run_calibration = (header["CAL_SIGNAL"] == "1")

            # run the transients processor
            # run_transients = (header["TRANSIENTS"] == "1")
            run_transients = False

            # RFI reference pol is assumed to be last pol
            if have_rfi_reference_pol:
                rfi_reference_pol = int(header["NPOL"]) - 1
                self.info("Header NPOL=" + str(int(header["NPOL"])) +
                          " RFI reference signal present in pol " +
                          str(rfi_reference_pol))
                cmd = cmd + " -r " + str(rfi_reference_pol)

            if run_adaptive_filter:
                self.info("Adaptive filter active")
                cmd = cmd + " -a "

            if run_calibration:
                self.info("Calibration active")
                try:
                    avg_time = header["TSYS_AVG_TIME"]
                except KeyError:
                    avg_time = "10"
                try:
                    freq_res = header["TSYS_FREQ_RES"]
                except KeyError:
                    freq_res = "1"
                cmd = cmd + " -c " + avg_time + " -e " + freq_res

            if run_transients:
                self.info("Transients active")
                cmd = cmd + " -f " + header["TRANS_TSAMP"]

            # AJ todo check the channelisation limits with Nuer
            if run_adaptive_filter or run_calibration or run_transients:
                cmd = cmd + " -n 1024"

            # create a log pipe for the stats command
            log_pipe = LogSocket("preproc_src", "preproc_src",
                                 str(self.id), "stream", log_host,
                                 log_port, int(DL))

            # connect up the log file output
            log_pipe.connect()

            # add this binary to the list of active commands
            self.binary_list.append("uwb_preprocessing_pipeline " + in_key)

            self.info("START " + cmd)

            # initialize the threads
            preproc_thread = preprocThread(cmd, proc_dir, log_pipe.sock, 2)

            self.debug("starting preproc thread")
            preproc_thread.start()
            self.debug("preproc thread started")

            self.debug("joining preproc thread")
            rval = preproc_thread.join()
            self.debug("preproc thread joined")

            self.info("END     " + cmd)

            if rval:
                self.error("preproc thread failed")
                cmd = "touch " + proc_dir + "/obs.finished"
                rval, lines = self.system(cmd, 2)
                self.quit_event.set()
            else:
                cmd = "touch " + proc_dir + "/obs.finished"
                rval, lines = self.system(cmd, 2)


###############################################################################


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print "ERROR: 1 command line argument expected"
        sys.exit(1)

    # this should come from command line argument
    stream_id = sys.argv[1]

    script = []
    script = PreprocessorDaemon("uwb_preprocessor", stream_id)

    state = script.configure(DAEMONIZE, DL, "preproc", "preproc")

    if state != 0:
        sys.exit(state)

    script.log(1, "STARTING SCRIPT")

    try:
        script.main()

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
