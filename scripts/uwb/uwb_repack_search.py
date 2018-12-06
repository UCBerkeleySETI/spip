#!/usr/bin/env python

##############################################################################
#
#     Copyright (C) 2015 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
#
###############################################################################
#
#     UWB Repack Search:
#       updates header parameters [PSRFITS]
#       copies output from a sub-band to web server
#

import os
import sys
import traceback
import time
import threading
from copy import deepcopy

from spip.daemons.bases import StreamBased
from spip.daemons.daemon import Daemon
from spip.threads.reporting_thread import ReportingThread
from spip.utils import sockets, times

DAEMONIZE = True
DL = 1


class RepackSearchReportingThread(ReportingThread):

    def __init__(self, script, id):
        host = sockets.getHostNameShort()
        port = int(script.cfg["BEAM_REPACK_SEARCH_PORT"])
        if id >= 0:
            port += int(id)
        ReportingThread.__init__(self, script, host, port)

        with open(script.cfg["WEB_DIR"] + "/spip/images/blankimage.gif",
                  mode='rb') as file:
            self.no_data = file.read()

    def parse_message(self, request):
        self.script.debug(str(request))

        xml = ""
        req = request["repack_search_request"]

        if req["type"] == "state":

            self.script.debug("preparing state response")
            xml = "<repack_search_state>"

            self.script.results["lock"].acquire()
            self.script.trace("acquired results lock")

            xml += "<stream id='" + str(self.script.id) + "' beam_name='" + \
                   self.script.beam_name + "' active='" + \
                   str(self.script.results["valid"]) + "'>"

            self.script.debug("keys="+str(self.script.results.keys()))

            if self.script.results["valid"]:

                self.script.trace("stream is valid")

                ts = str(self.script.results["timestamp"])
                xml += "<plot type='timeseries' timestamp='" + ts + "'/>"
                xml += "<plot type='freqtime' timestamp='" + ts + "'/>"
                xml += "<plot type='histogram' timestamp='" + ts + "'/>"

            xml += "</stream>"

            self.script.results["lock"].release()
            self.script.trace("released results lock")

            xml += "</repack_search_state>"
            self.script.debug("returning " + str(xml))

            return True, xml + "\r\n"

        elif req["type"] == "plot":

            self.script.debug("req[plot]=" + req["plot"] + " valid_plots=" + str(self.script.valid_plots))
            if req["plot"] in self.script.valid_plots:

                self.script.results["lock"].acquire()
                self.script.debug("plot=" + req["plot"])

                if self.script.results["valid"]:
                    plot = req["plot"] + "_" + req["res"]

                    if plot in self.script.results.keys():
                        bin_data_len = len(self.script.results[plot])
                        if bin_data_len > 64:
                            bin_data = deepcopy(self.script.results[plot])
                            self.script.trace(plot + " valid, image len=" +
                                              str(bin_data_len))
                            self.script.results["lock"].release()
                            return False, bin_data
                        else:
                            self.script.debug("image length=" +
                                              str(bin_data_len) + " <= 64")
                    else:
                        self.script.debug("plot ["+plot+"] not in keys [" +
                                          str(self.script.results.keys()))
                else:
                    self.script.debug("results not valid")

                # return empty plot
                self.script.debug("[returning NO DATA YET]")
                self.script.results["lock"].release()
                return False, self.no_data

            else:

                self.script.info("invalid plot, " + req["plot"] + " not in " +
                                 str(self.script.valid_plots))
                return False, self.no_data
        else:
            self.script.debug("unrecognized req type: " + req["type"])

        xml += "<preproc_state>"
        xml += "<error>Invalid request</error>"
        xml += "</preproc_state>\r\n"

        return True, xml


class RepackSearchDaemon(Daemon):

    # RepackSearchDaemon::__init__
    def __init__(self, name, id):
        Daemon.__init__(self, name, str(id))

        self.beam = ""
        self.subbands = []
        self.results = {}
        self.results["lock"] = threading.Lock()
        self.results["valid"] = False
        self.beam_name = ""
        self.valid_plots = ["timeseries", "histogram", "freqtime"]

    # RepackSearchDaemon::main
    def main(self):

        files_glob = "????-??-??-??:??:??.?f"

        self.debug("beam=" + str(self.beam))

        if not os.path.exists(self.processing_dir):
            os.makedirs(self.processing_dir, 0755)
        if not os.path.exists(self.finished_dir):
            os.makedirs(self.finished_dir, 0755)
        if not os.path.exists(self.send_dir):
            os.makedirs(self.send_dir, 0755)

        self.debug("stream_id=" + str(self.id))

        while (not self.quit_event.isSet()):

            processed_this_loop = 0

            # check each beam for searched archives to process
            beam_dir = self.processing_dir + "/" + self.beam
            self.debug("beam=" + self.beam + " beam_dir=" + beam_dir)

            if not os.path.exists(beam_dir):
                os.makedirs(beam_dir, 0755)

            # get a list of all the processing observations for any utc/source
            cmd = "find " + beam_dir + " -mindepth 3 -maxdepth 3 -type d " + \
                  "-name '" + self.cfreq + "'"
            rval, paths = self.system(cmd, 2)

            for path in paths:

                # strip prefix
                observation = path[(len(beam_dir)+1):]

                parts = observation.split("/")
                if len(parts) != 3:
                    self.warn("path=" + path + " observation=" + observation +
                              " did not have 3 components")
                    continue

                (utc, source, cfreq) = observation.split("/")
                self.debug("found processing obs utc=" + utc + " source=" +
                           source + " cfreq=" + cfreq)

                # skip any stats sources
                if source == "stats":
                    continue

                in_dir = beam_dir + "/" + observation

                # find searched archives in the processing directory
                cmd = "find " + in_dir + " -mindepth 1 -maxdepth 1 " + \
                      "-type f -name '" + files_glob + "' -printf '%f\\n' " +\
                      "| sort -n"
                rval, files = self.system(cmd, 3)
                self.debug("utc=" + utc + " had " + str(len(files)) + " files")

                # processing in chronological order
                files.sort()

                for file in files:

                    self.log(1, observation + ": processing " + file)

                    # get the most recent file
                    is_most_recent = True

                    self.debug("process_archive("+utc+", "+source+", "+file)
                    (rval, processed, response) = self.process_archive(utc, source, file,
                                                            is_most_recent)
                    processed_this_loop += processed
                    if rval:
                        self.warn("failed to process " + file + " " + response)

                # check if the observation has been marked as failed
                filename = in_dir + "/obs.failed"
                if os.path.exists(filename):
                    self.info(observation + ": processing -> failed")

                    self.debug("fail_observation ("+utc+","+source+")")
                    (rval, response) = self.fail_observation(utc, source)
                    if rval:
                        self.warn("failed to finalise: " + response)

                    # terminate this loop
                    continue

                # check if the observation has been marked as finished
                filename = in_dir + "/obs.finished"
                if os.path.exists(filename) and processed_this_loop == 0:
                    # if the file was created at least 20s ago, then there are
                    # unlikely to be any more sub-ints to process
                    self.debug(observation + " mtime=" +
                               str(os.path.getmtime(filename)) +
                               " time=" + str(time.time()))
                    if os.path.getmtime(filename) + 20 < time.time():
                        self.info(observation + ": processing -> finished")
                        self.debug("finalise_observation("+utc+","+source+")")
                        (rval, response) = self.finalise_observation(utc,
                                                                     source)
                        if rval:
                            self.warn("failed to finalise: " + response)

            if processed_this_loop == 0:
                to_sleep = 5
                while to_sleep > 0 and not self.quit_event.isSet():
                    self.trace("sleep(1)")
                    time.sleep(1)
                    to_sleep -= 1

    # convert a UTC in ????-??-??-??:??:?? to ????-??-??-??-??-??
    def atnf_utc(self, instr):
        return instr.replace(":", "-")

    # patch missing information into the PSRFITS header
    def patch_psrfits_header(self, input_dir, input_file):

        header_file = input_dir + "/obs.header"
        self.trace("header_file="+header_file)
        # header = Config.readCFGFileIntoDict (input_dir + "/obs.header")
        return (0, "")

    # create a remote directory for the observation
    def create_remote_dir(self, utc, source):

        beam_utc_source = self.beam + "/" + utc + "/" + source
        in_dir = self.processing_dir + "/" + beam_utc_source + "/" + self.cfreq
        rem_dir = self.cfg["SERVER_SEARCH_DIR"] + "/processing/" + \
            beam_utc_source + "/" + self.cfreq
        ctrl_file = in_dir + "/remdir.created"

        # use a local control file to know if remote directory exists
        if not os.path.exists(ctrl_file):
            cmd = "ssh " + self.rsync_user + "@" + self.rsync_server + \
                " 'mkdir -p " + rem_dir + "'"
            rval, lines = self.system(cmd, 2)
            if rval:
                return (rval, "failed to create remote dir " + str(lines[0]))

            cmd = "touch " + ctrl_file
            rval, lines = self.system(cmd, 2)
            if rval:
                return (rval, "failed to create control file " + str(lines[0]))

            # copy the header to the remote directory
            header_file = in_dir + "/obs.header"
            cmd = "rsync -a " + header_file + " " + self.rsync_user + "@" + \
                self.rsync_server + "::" + self.rsync_module + \
                "/search/processing/" + self.beam + "/" + utc + \
                "/" + source + "/" + self.cfreq + "/"

            # we dont have an rsync module setup on the server yet
            cmd = "rsync -a " + header_file + " " + self.rsync_user + "@" + \
                self.rsync_server + ":" + self.cfg["SERVER_SEARCH_DIR"] + \
                "/processing/" + self.beam + "/" + utc + \
                "/" + source + "/" + self.cfreq + "/"

            rval, lines = self.system(cmd, 2)
            if rval:
                return (rval, "failed to rsync files to the server")

            return (0, "remote directory created")
        else:
            return (0, "remote directory existed")

    # process and file in the directory, adding file to
    def process_archive(self, utc, source, file, plot):

        utc_out = self.atnf_utc(utc)
        in_dir = self.processing_dir + "/" + self.beam + "/" + utc + "/" + \
            source + "/" + self.cfreq
        out_dir = self.send_dir + "/" + self.beam + "/" + utc_out + "/" + \
            source + "/" + self.cfreq
        processed = 0

        # ensure required directories exist
        required_dirs = [out_dir]
        for dir in required_dirs:
            if not os.path.exists(dir):
                os.makedirs(dir, 0755)

        # create a remote directory on the server
        rval, message = self.create_remote_dir(utc, source)
        if rval:
            self.error("create_remote_dir failed: " + message)
            return (rval, processed, "failed to create remote directory: " +
                    message)

        input_file = in_dir + "/" + file
        header_file = in_dir + "/obs.header"

        self.debug("input_file=" + input_file + " header_file=" + header_file)
        if not os.path.exists(out_dir + "/obs.header"):
            cmd = "cp " + in_dir + "/obs.header " + out_dir + "/obs.header"
            rval, lines = self.system(cmd, 2)
            if rval:
                return (rval, "failed to copy obs.header to out_dir: " +
                        str(lines[0]))

        if plot:
            rval, response = self.plot_archive(utc, source, input_file)
            if rval:
                self.debug("plot_archive failed: " + response)
                time.sleep(1)
            else:
                processed = 1

        # rename the psrfits file to match ATNF standards
        atnf_psrfits_file = out_dir + "/" + self.atnf_utc(
                os.path.basename(input_file))
        try:
            os.rename(input_file, atnf_psrfits_file)
            processed = 1
        except OSError, e:
            self.warn("fail_observation: failed to rename " + input_file +
                      " to " + atnf_psrfits_file + ":" + str(e))
            return (1, "failed to rename psrfits file")

        return (0, processed, "")

    # generate monitoring plots for the archive
    def plot_archive(self, utc, source, file):

        beam_utc_source = self.beam + "/" + utc + "/" + source
        dir = self.processing_dir + "/" + beam_utc_source + "/" + self.cfreq

        # command will generate 3 files
        os.chdir(dir)
        cmd = "pfits_plotsearch -f " + file
        rval, lines = self.system(cmd, 3)
        if rval:
            # return to root directory
            os.chdir("/")
            return (rval, "failed to generate plots for " + file)

        # acquire the results lock
        self.trace("acquiring lock")
        self.results["lock"].acquire()

        self.debug("writing image files to results")
        plot_types = ["freqtime", "histogram", "timeseries"]
        resolutions = {"1024x768": "hi", "300x225": "lo"}
        count = 0
        for t in plot_types:
            for r in resolutions.keys():
                filename = "pfits_" + t + "." + r + ".png"
                if os.path.exists(filename):
                    fptr = open(filename, "rb")
                    self.results[t + "_" + resolutions[r]] = fptr.read()
                    fptr.close()
                    os.remove(filename)
                    count += 1

        self.results["valid"] = (count == len(plot_types) *
                                 len(resolutions.keys()))
        self.results["timestamp"] = times.getCurrentTime()
        self.trace("image timestamp=" + self.results["timestamp"])
        self.results["lock"].release()
        self.trace("released lock")

        # return to root directory
        os.chdir("/")

        return (0, "")

    # transition observation from processing to finished
    def finalise_observation(self, utc, source):

        utc_out = self.atnf_utc(utc)

        beam_utc_source = self.beam + "/" + utc + "/" + source
        beam_utcout_source = self.beam + "/" + utc_out + "/" + source

        in_dir = self.processing_dir + "/" + beam_utc_source + "/" + self.cfreq
        fin_dir = self.finished_dir + "/" + beam_utc_source + "/" + self.cfreq
        send_dir = self.send_dir + "/" + beam_utcout_source + "/" + self.cfreq

        # ensure required directories exist
        required_dirs = [fin_dir, send_dir]
        for dir in required_dirs:
            if not os.path.exists(dir):
                os.makedirs(dir, 0755)

        # touch and obs.finished file in the archival directory
        cmd = "touch " + send_dir + "/obs.finished"
        rval, lines = self.system(cmd, 3)
        if rval:
            return (1, "failed to create " + send_dir + "/obs.finished")

        # create remote directory on server
        self.trace("create_remote_dir(" + utc + ", " + source + ")")
        (rval, message) = self.create_remote_dir(utc, source)
        if rval:
            return (1, "failed to create remote dir: " + message)

        # touch obs.finished file in the remote directory
        rem_cmd = "touch " + self.cfg["SERVER_SEARCH_DIR"] + "/processing/" + \
                  self.beam + "/" + utc + "/" + source + "/" + self.cfreq + \
                  "/obs.finished"
        cmd = "ssh " + self.rsync_user + "@" + self.rsync_server + " '" + \
              rem_cmd + "'"
        rval, lines = self.system(cmd, 2)
        if rval:
            return (rval, "failed to touch remote obs.finished file: " +
                    str(lines[0]))

        # simply move the observation from processing to finished
        try:
            self.trace("rename(" + in_dir + "," + fin_dir + ")")
            os.rename(in_dir, fin_dir)
        except OSError, e:
            self.warn("failed to rename in_dir: " + str(e))
            return (1, "failed to rename in_dir to fin_dir")

        # count the number of other sub-band directories
        src_dir = os.path.dirname(in_dir)
        self.debug("src_dir=" + src_dir + " list=" + str(os.listdir(src_dir)))
        subdir_count = len(os.listdir(src_dir))
        self.debug("src_dir subdir_count=" + str(subdir_count))
        if subdir_count == 0:
            try:
                os.rmdir(src_dir)
            except OSError:
                self.debug(src_dir + " was deleted by another stream")
            utc_dir = os.path.dirname(src_dir)
            try:
                os.rmdir(utc_dir)
            except OSError:
                self.debug(utc_dir + " was deleted by another stream")

        return (0, "")

    # transition an observation from processing to failed
    def fail_observation(self, utc, source):

        utc_out = self.atnf_utc(utc)
        beam_utc_source_cfreq = self.beam + "/" + utc + "/" + source + "/" + \
            self.cfreq
        beam_utcout_source_cfreq = self.beam + "/" + utc_out + "/" + source + \
            "/" + self.cfreq

        in_dir = self.processing_dir + "/" + beam_utc_source_cfreq
        fail_dir = self.failed_dir + "/" + beam_utc_source_cfreq
        send_dir = self.send_dir + "/" + beam_utcout_source_cfreq

        # ensure required directories exist
        required_dirs = [fail_dir, send_dir]
        for dir in required_dirs:
            if not os.path.exists(dir):
                os.makedirs(dir, 0755)

        # touch obs.failed file in the send directory
        cmd = "touch " + send_dir + "/obs.failed"
        rval, lines = self.system(cmd, 3)

        # create remote directory on the server
        self.trace("create_remote_dir(" + utc + ", " + source + ")")
        (rval, message) = self.create_remote_dir(utc, source)
        if rval:
            return (1, "failed to create remote directory: " + message)

        # touch obs.failed file in the remote processing directory
        rem_cmd = "touch " + self.cfg["SERVER_SEARCH_DIR"] + "/processing/" + \
            beam_utc_source_cfreq + "/obs.failed"
        cmd = "ssh " + self.rsync_user + "@" + self.rsync_server + " '" + \
            rem_cmd + "'"
        rval, lines = self.system(cmd, 2)
        if rval:
            return (rval, "failed to touch remote obs.failed file")

        # simply move the observation to the failed directory
        try:
            fail_parent_dir = os.path.dirname(fail_dir)
            if not os.path.exists(fail_parent_dir):
                os.mkdir(fail_parent_dir, 0755)
            os.rename(in_dir, fail_dir)
        except OSError, e:
            self.info("failed to rename " + in_dir + " to " + fail_dir)
            self.info(str(e))
            return (1, "failed to rename in_dir to fail_dir")

        # count the number of other sub-band directories
        src_dir = os.path.dirname(in_dir)
        self.debug("src_dir=" + src_dir)
        subdir_count = len(os.listdir(src_dir))
        self.debug("src_dir subdir_count=" + str(subdir_count))
        if subdir_count == 0:
            try:
                os.rmdir(src_dir)
            except OSError:
                self.debug(src_dir + " was deleted by another stream")
            utc_dir = os.path.dirname(src_dir)
            try:
                os.rmdir(utc_dir)
            except OSError:
                self.debug(utc_dir + " was deleted by another stream")

        return (0, "")


class RepackSearchStreamDaemon (RepackSearchDaemon, StreamBased):

    # RepackSearchStreamDaemon::__init__
    def __init__(self, name, id):
        RepackSearchDaemon.__init__(self, name, str(id))
        StreamBased.__init__(self, str(id), self.cfg)

    # RepackSearchStreamDaemon::configure
    def configure(self, become_daemon, dl, source, dest):

        self.debug("")
        Daemon.configure(self, become_daemon, dl, source, dest)

        self.trace("stream_id=" + self.id)
        self.trace("beam_id=" + self.beam_id)
        self.trace("subband_id=" + self.subband_id)

        # beam_name
        self.beam = self.cfg["BEAM_" + str(self.beam_id)]

        # base directories for search data products
        self.processing_dir = self.cfg["CLIENT_SEARCH_DIR"] + "/processing"
        self.finished_dir = self.cfg["CLIENT_SEARCH_DIR"] + "/finished"
        self.send_dir = self.cfg["CLIENT_SEARCH_DIR"] + "/send"
        self.failed_dir = self.cfg["CLIENT_SEARCH_DIR"] + "/failed"

        # get the properties for this subband
        (cfreq, bw, nchan) = self.cfg["SUBBAND_CONFIG_" +
                                      str(self.subband_id)].split(":")
        self.cfreq = cfreq
        self.bw = bw
        self.nchan = nchan

        # not sure is this is needed
        self.out_cfreq = cfreq

        # Rsync parameters
        self.rsync_user = "uwb"
        self.rsync_server = "medusa-srv0.atnf.csiro.au"
        self.rsync_module = "TBD"

        # Beam Name
        (host, beam_id, subband_id) = self.cfg["STREAM_" + self.id].split(":")
        self.beam_name = self.cfg["BEAM_" + beam_id]


        return 0


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print "ERROR: 1 command line argument expected"
        sys.exit(1)

    # this should come from command line argument
    stream_id = sys.argv[1]

    if int(stream_id) < 0:
        print "ERROR: RepackSearchDaemon can only operate on streams"
        sys.exit(1)
    else:
        script = RepackSearchStreamDaemon("uwb_repack_search", stream_id)

    state = script.configure(DAEMONIZE, DL, "repack_search", "repack_search")
    if state != 0:
        script.quit_event.set()
        sys.exit(state)

    script.log(1, "STARTING SCRIPT")

    try:


        reporting_thread = RepackSearchReportingThread(script, stream_id)
        reporting_thread.start()

        script.main ()

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
