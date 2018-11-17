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
import time
import threading
import copy
import numpy as np

from spip.daemons.bases import StreamBased
from spip.daemons.daemon import Daemon
from spip.threads.reporting_thread import ReportingThread
from spip.utils import times, sockets
from spip.config import Config
from spip.plotting import BandpassPlot, TimeseriesPlot

DAEMONIZE = True
DL = 1


class PreprocessorReportingThread(ReportingThread):

    def __init__(self, script, id):
        host = sockets.getHostNameShort()
        port = int(script.cfg["STREAM_PREPROC_PORT"])
        if id >= 0:
            port += int(id)
        ReportingThread.__init__(self, script, host, port)

        with open(script.cfg["WEB_DIR"] + "/spip/images/blankimage.gif",
                  mode='rb') as file:
            self.no_data = file.read()

    def parse_message(self, request):
        self.script.debug(str(request))

        xml = ""
        req = request["preproc_request"]

        if req["type"] == "state":

            self.script.trace("preparing state response")
            xml = "<preproc_state>"

            self.script.results["lock"].acquire()
            xml += "<stream id='" + str(self.script.id) + "' beam_name='" + \
                self.script.beam_name + "' active='" + \
                str(self.script.results["valid"]) + "'>"

            self.script.trace("keys="+str(self.script.results.keys()))

            if self.script.results["valid"]:
                self.script.trace("stream is valid!")

                ts = str(self.script.results["timestamp"])
                xml += "<plot type='cleaned' timestamp='" + ts + "'/>"
                xml += "<plot type='dirty' timestamp='" + ts + "'/>"
                xml += "<plot type='gainsfreq' timestamp='" + ts + "'/>"
                xml += "<plot type='gainstime' timestamp='" + ts + "'/>"

            xml += "</stream>"

            self.script.results["lock"].release()

            xml += "</preproc_state>"
            self.script.debug("returning " + str(xml))

            return True, xml + "\r\n"

        elif req["type"] == "plot":

            if req["plot"] in self.script.valid_plots:

                self.script.results["lock"].acquire()
                self.script.debug("plot=" + req["plot"] + " res=" + req["res"])

                if self.script.results["valid"]:
                    plot = req["plot"] + "_" + req["res"]

                    self.script.debug("plot=" + plot)
                    if plot in self.script.results.keys():
                        plot_len = len(self.script.results[plot])
                        if plot_len > 64:
                            bin_data = copy.deepcopy(self.script.results[plot])
                            self.script.trace("" + plot + " valid, image " +
                                              "len=" + str(len(bin_data)))
                            self.script.results["lock"].release()
                            return False, bin_data
                        else:
                            self.script.info("image length=" + str(plot_len) +
                                             " <= 64")
                    else:
                        self.script.info("plot ["+plot+"] not in keys [" +
                                         str(self.script.results.keys()))
                else:
                    self.script.info("results not valid")

                # return empty plot
                self.script.info("[returning NO DATA YET]")
                self.script.results["lock"].release()
                return False, self.no_data

            else:

                self.script.info("invalid plot, " + req["plot"] + " not in " +
                                 str(self.script.valid_plots))
                self.script.info("returning 'no_data' of size " +
                                 str(len(self.no_data)))
                return False, self.no_data

        else:

            xml += "<preproc_state>"
            xml += "<error>Invalid request</error>"
            xml += "</preproc_state>\r\n"

            return True, xml


class RepackPreprocessorDaemon(Daemon, StreamBased):

    def __init__(self, name, id):
        Daemon.__init__(self, name, str(id))
        StreamBased.__init__(self, id, self.cfg)

        self.valid_plots = []
        self.results = {}

        self.results["lock"] = threading.Lock()
        self.results["valid"] = False

        (host, beam_id, sub_id) = self.cfg["STREAM_" + id].split(":")
        self.beam = self.cfg["BEAM_" + beam_id]

        (cfreq, bw, nchan) = self.cfg["SUBBAND_CONFIG_" + sub_id].split(":")
        self.cfreq = cfreq

        self.proc_dir = self.cfg["CLIENT_PREPROC_DIR"] + "/processing/" + \
            self.beam
        self.fin_dir = self.cfg["CLIENT_PREPROC_DIR"] + "/finished/" + \
            self.beam
        self.send_dir = self.cfg["CLIENT_PREPROC_DIR"] + "/send/" + \
            self.beam
        self.fail_dir = self.cfg["CLIENT_PREPROC_DIR"] + "/failed/" + \
            self.beam

        self.dirty_valid = False
        self.cleaned_valid = False
        self.gains_valid = False

    #################################################################
    # main
    #             id >= 0     process folded archives from a stream
    #             id == -1    process folded archives from all streams
    def main(self):

        self.dirty_plot = BandpassPlot()
        self.cleaned_plot = BandpassPlot()
        self.gains_time_plot = TimeseriesPlot()
        self.gains_freq_plot = BandpassPlot()

        self.valid_plots = ["dirty", "cleaned", "gainstime", "gainsfreq"]

        required_dirs = [self.proc_dir, self.fin_dir, self.send_dir,
                         self.fail_dir]
        for dir in required_dirs:
            if not os.path.exists(dir):
                os.makedirs(dir, 0755)

        # get the data block keys
        stream_id = str(self.id)
        self.debug("stream_id=" + str(self.id))

        log = False
        zap = False
        transpose = False

        # configure the bandpass plots
        self.dirty_plot.configure(log, zap, transpose)
        self.cleaned_plot.configure(log, zap, transpose)
        self.gains_time_plot.setLabels("Adaptive Filter Gain vs Time",
                                       "Time (seconds)",
                                       "Maximum Filter Amplitude")
        self.gains_freq_plot.setLabels("Adaptive Filter Gain vs Frequency",
                                       "Frequency (MHz)",
                                       "Maximum Filter Amplitude")

        # determine the number of channels to be processed by this stream
        (cfreq, bw, nchan) = self.cfg["SUBBAND_CONFIG_" + stream_id].split(":")

        # enter the main loop
        while (not self.quit_event.isSet()):

            self.debug("main processing loop")
            processed_this_loop = 0

            # get a list of all the processing observations for any utc/source
            cmd = "find " + self.proc_dir + " -mindepth 3 -maxdepth 3 " + \
                  "-type d -name '" + self.cfreq + "' | sort -n"
            rval, paths = self.system(cmd, 3)
            if rval:
                self.warn("could not look for files in beam_dir")
                time.sleep(5)
                continue

            self.debug("found " + str(len(paths)) + " processing observations")

            # look at all observations marked processing
            for path in paths:

                # short circuit
                if self.quit_event.isSet():
                    continue

                # strip prefix
                observation = path[(len(self.proc_dir)+1):]

                parts = observation.split("/")
                if len(parts) != 3:
                    self.warn("path=" + path + " observation=" + observation +
                              " did not have 3 components")
                    continue

                (utc, source, cfreq) = observation.split("/")
                self.debug("utc=" + utc + " source=" + source + " cfreq=" +
                           cfreq)

                proc_dir = self.proc_dir + "/" + observation

                # get a listing of all the files in this directory
                cmd = "find " + proc_dir + " -mindepth 1 -maxdepth 1 " + \
                      "-type f  -printf '%f\\n' " + "| sort -n"
                rval, files = self.system(cmd, 2)
                self.debug("utc=" + utc + " had " + str(len(files)) + " files")

                # processing in chronological order
                files.sort()

                # antf preferred UTC
                utc_out = self.atnf_utc(utc)

                # directory in which to run preprocessor
                fin_dir = self.fin_dir + "/" + utc + "/" + source + \
                    "/" + cfreq
                send_dir = self.send_dir + "/" + utc_out + "/" + source + \
                    "/" + cfreq

                if not os.path.exists(fin_dir):
                    os.makedirs(fin_dir, 0755)
                if not os.path.exists(send_dir):
                    os.makedirs(send_dir, 0755)

                # get a list of all the files in proc_dir, including obs.header
                files = os.listdir(proc_dir)
                self.debug("files="+str(files))
                if len(files) == 0:
                    continue

                # only proceed if an obs.header file existed
                cmd = "ls -l " + proc_dir
                rval, files = self.system(cmd, 2)
                if not os.path.exists(proc_dir + "/obs.header"):
                    self.warn("obs.header file did not exist " + proc_dir +
                              "/obs.header")
                    continue

                # ensure obs.header exists in the send dir
                if not os.path.exists(send_dir + "/obs.header"):
                    cmd = "cp " + proc_dir + "/obs.header " + send_dir + \
                          "/obs.header"
                    rval, lines = self.system(cmd, 2)
                    if rval:
                        return (rval, "failed to copy obs.header to "
                                "send_dir: " + str(lines[0]))

                # look for output from the adaptive filter
                self.debug("processing adaptive filter output")
                cleaned_processed = self.process_cleaned(proc_dir, send_dir)
                dirty_processed = self.process_dirty(proc_dir, send_dir)
                gains_processed = self.process_gains(proc_dir, send_dir)
                adaptive_filter_processed = cleaned_processed + \
                    dirty_processed + \
                    gains_processed
                self.debug("processed " + str(adaptive_filter_processed) +
                           " files from adaptive filter")

                self.results["lock"].acquire()
                self.results["timestamp"] = times.getCurrentTime()
                self.results["valid"] = self.cleaned_valid and \
                    self.dirty_valid and \
                    self.gains_valid
                self.results["lock"].release()

                self.debug("processing calibration output")
                cal_processed = self.process_calibration(proc_dir, send_dir)
                self.debug("processed " + str(cal_processed) +
                           " files from calibration")

                # total number of input files processed this iteration
                processed_this_obs = adaptive_filter_processed + cal_processed

                filename = proc_dir + "/obs.failed"
                if os.path.exists(filename):
                    self.log(1, observation + ": processing -> failed")

                    self.debug("fail_observation ("+utc+","+source+")")
                    (rval, response) = self.fail_observation(utc, source)
                    if rval:
                        self.warn("failed to finalise: " + response)

                    # terminate this loop
                    continue

                # check if the observation has been marked as finished
                filename = proc_dir + "/obs.finished"
                if os.path.exists(filename) and processed_this_obs == 0:
                    # if the file was created at least 20s ago, then there are
                    # unlikely to be any more sub-ints to process
                    self.debug(observation + " mtime=" +
                               str(os.path.getmtime(filename)) +
                               " time=" + str(time.time()))
                    if os.path.getmtime(filename) + 20 < time.time():
                        self.log(1, observation + ": processing -> finished")
                        self.debug("finalise_observation("+utc+","+source+")")
                        (rval, response) = self.finalise_observation(utc,
                                                                     source)
                        if rval:
                            self.warn("failed to finalise: " + response)

                processed_this_loop += processed_this_obs

            if processed_this_loop == 0:
                to_sleep = 5
                while to_sleep > 0 and not self.quit_event.isSet():
                    self.trace("sleep(1)")
                    time.sleep(1)
                    to_sleep -= 1

    # convert a UTC in ????-??-??-??:??:?? to ????-??-??-??-??-??
    def atnf_utc(self, instr):
        return instr.replace(":", "-")

    # transition an observation from processing to failed
    def fail_observation(self, utc, source):

        utc_out = self.atnf_utc(utc)
        utc_source_cfreq = utc + "/" + source + "/" + self.cfreq
        utcout_source_cfreq = utc_out + "/" + source + "/" + self.cfreq
        in_dir = self.proc_dir + "/" + utc_source_cfreq
        fail_dir = self.fail_dir + "/" + utc_source_cfreq
        send_dir = self.send_dir + "/" + utcout_source_cfreq

        # ensure required directories exist
        required_dirs = [fail_dir, send_dir]
        for dir in required_dirs:
            if not os.path.exists(dir):
                os.makedirs(dir, 0755)

        # touch obs.failed file in the send directory
        cmd = "touch " + send_dir + "/obs.failed"
        rval, lines = self.system(cmd, 3)

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
            os.rmdir(src_dir)
            utc_dir = os.path.dirname(src_dir)
            os.rmdir(utc_dir)

        return (0, "")

    # transition observation from processing to finished
    def finalise_observation(self, utc, source):

        utc_out = self.atnf_utc(utc)

        utc_source_cfreq = utc + "/" + source + "/" + self.cfreq
        utcout_source_cfreq = utc_out + "/" + source + "/" + self.cfreq

        in_dir = self.proc_dir + "/" + utc_source_cfreq
        fin_dir = self.fin_dir + "/" + utc_source_cfreq
        send_dir = self.send_dir + "/" + utcout_source_cfreq

        # ensure required directories exist
        required_dirs = [fin_dir, send_dir]
        for dir in required_dirs:
            if not os.path.exists(dir):
                os.makedirs(dir, 0755)

        # touch and obs.finished file in the archival directory
        cmd = "touch " + send_dir + "/obs.finished"
        rval, lines = self.system(cmd, 2)
        if rval:
            return (1, "failed to create " + send_dir + "/obs.finished")

        # simply move the observation from processing to finished
        try:
            self.debug("rename(" + in_dir + "," + fin_dir + ")")
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
            os.rmdir(src_dir)
            utc_dir = os.path.dirname(src_dir)
            os.rmdir(utc_dir)

        return (0, "")

    def process_dirty(self, proc_dir, send_dir):

        # find the most recent HG stats file
        files = [file for file in os.listdir(proc_dir)
                 if file.lower().endswith(".dirty")]
        self.trace("files=" + str(files))

        if len(files) > 0:

            # process only the most recent file
            last_file = proc_dir + "/" + files[-1]
            self.debug("procesing " + last_file)

            # read the data from file into a numpy array
            file_size = os.path.getsize(last_file)
            header_size = 4096
            data_size = file_size - header_size

            # read the data from file into a numpy array
            fptr = open(last_file, "rb")
            header_str = fptr.read(header_size)
            header = Config.readDictFromString(header_str)

            npol = int(header["NPOL"])
            ndim = int(header["NDIM"])
            nchan = int(header["NCHAN"])
            nant = int(header["NANT"])
            nbit = int(header["NBIT"])
            freq = float(header["FREQ"])
            bw = float(header["BW"])

            bytes_per_sample = (npol * ndim * nchan * nant * nbit) / 8
            ndat = data_size / bytes_per_sample

            # TODO check that nbit==32 ndat==1 nant==1
            nval = ndat*nant*nchan*npol*ndim

            raw = np.fromfile(fptr, dtype=np.float32, count=nval)
            fptr.close()

            # reshape the raw data into a numpy array with specified dimensions
            data = raw.reshape((npol, nchan))

            if npol == 1:
                labels = ["AA"]
            elif npol == 2:
                labels = ["AA", "BB"]
            elif npol == 3:
                labels = ["AA", "BB", "REF"]
            else:
                labels = []

            # acquire the results lock
            self.results["lock"].acquire()

            # generate plots multi polarisation plots
            self.dirty_plot.plot_npol(240, 180, True, nchan, freq, bw, data,
                                      labels)
            self.results["dirty_lo"] = self.dirty_plot.getRawImage()

            self.dirty_plot.plot_npol(1024, 768, False, nchan, freq, bw, data,
                                      labels)
            self.results["dirty_hi"] = self.dirty_plot.getRawImage()

            self.dirty_valid = True
            self.results["lock"].release()

            for file in files:
                os.rename(proc_dir + "/" + file, send_dir + "/" + file)

        return len(files)

    def process_cleaned(self, proc_dir, send_dir):

        files = [file for file in os.listdir(proc_dir)
                 if file.lower().endswith(".cleaned")]
        self.trace("files=" + str(files))

        if len(files) > 0:

            # process only the most recent file
            last_file = proc_dir + "/" + files[-1]
            self.info("file=" + last_file)

            # read the data from file into a numpy array
            file_size = os.path.getsize(last_file)
            header_size = 4096
            data_size = file_size - header_size

            # read the data from file into a numpy array
            fptr = open(last_file, "rb")
            header_str = fptr.read(header_size)
            header = Config.readDictFromString(header_str)

            npol = int(header["NPOL"])
            ndim = int(header["NDIM"])
            nchan = int(header["NCHAN"])
            nant = int(header["NANT"])
            nbit = int(header["NBIT"])
            freq = float(header["FREQ"])
            bw = float(header["BW"])
            self.info("npol=" + str(npol) + " ndim=" + str(ndim) + " nchan=" +
                      str(nchan) + " nant=" + str(nant) + " nbit=" +
                      str(nbit) + " freq=" + str(freq) + " bw=" + str(bw))

            bytes_per_sample = (npol * ndim * nchan * nant * nbit) / 8
            ndat = data_size / bytes_per_sample

            # TODO check that nbit==32 ndat==1 nant=1
            nval = ndat*nant*nchan*npol*ndim
            self.info("bytes_per_sample=" + str(bytes_per_sample) + " nval=" +
                      str(nval))

            raw = np.fromfile(fptr, dtype=np.float32, count=nval)
            fptr.close()

            # reshape the raw data into a numpy array with specified dimensions
            self.info("np.shape(raw)=" + str(np.shape(raw)))
            self.info("npol=" + str(npol) + " nchan=" + str(nchan))
            data = raw.reshape((npol, nchan))

            if npol == 1:
                labels = ["AA"]
            elif npol == 2:
                labels = ["AA", "BB"]
            else:
                labels = []

            # acquire the results lock
            self.results["lock"].acquire()

            self.info("len(data)=" + str(len(data)))

            # generate plots multi polarisation plots
            self.cleaned_plot.plot_npol(240, 180, True, nchan, freq, bw,
                                        data, labels)
            self.results["cleaned_lo"] = self.cleaned_plot.getRawImage()

            self.cleaned_plot.plot_npol(1024, 768, False, nchan, freq, bw,
                                        data, labels)
            self.results["cleaned_hi"] = self.cleaned_plot.getRawImage()
            self.cleaned_valid = True
            self.results["lock"].release()

            for file in files:
                os.rename(proc_dir + "/" + file, send_dir + "/" + file)
        return len(files)

    def process_gains(self, proc_dir, send_dir):

        # find the most recent gains file
        files = [file for file in os.listdir(proc_dir)
                 if file.lower().endswith(".gains")]
        self.trace("files=" + str(files))

        if len(files) > 0:

            gains_time_file = proc_dir + "/gains.time"
            cmd = ""

            # combine all the gains files together
            if os.path.exists(gains_time_file):
                cmd = "uwb_adaptive_filter_tappend " + gains_time_file
                for file in files:
                    cmd = cmd + " " + proc_dir + "/" + file
                cmd = cmd + " " + gains_time_file
            else:
                if len(files) == 1:
                    cmd = "cp " + proc_dir + "/" + files[0] + " " + \
                        gains_time_file
                else:
                    cmd = "uwb_adaptive_filter_tappend"
                    for file in files:
                        cmd = cmd + " " + proc_dir + "/" + file
                    cmd = cmd + " " + gains_time_file

            self.info(cmd)
            rval, lines = self.system(cmd, 2)
            if not rval == 0:
                self.warn("failed to tappend gains files")
                return

            # read the data from file into a numpy array
            file_size = os.path.getsize(gains_time_file)
            header_size = 4096
            data_size = file_size - header_size

            gains_file = open(gains_time_file, "rb")
            header_str = gains_file.read(header_size)
            header = Config.readDictFromString(header_str)

            npol = int(header["NPOL"])
            ndim = int(header["NDIM"])
            nchan = int(header["NCHAN"])
            nant = int(header["NANT"])
            nbit = int(header["NBIT"])
            freq = float(header["FREQ"])
            bw = float(header["BW"])
            tsamp = float(header["TSAMP"])

            self.info("npol=" + str(npol) + " ndim=" + str(ndim) + " nchan=" +
                      str(nchan) + " nant=" + str(nant) + " nbit=" +
                      str(nbit) + " freq=" + str(freq) + " bw=" + str(bw))

            # check that the nbit is 32
            bytes_per_sample = (npol * ndim * nchan * nant * nbit) / 8
            ndat = data_size / bytes_per_sample
            nval = ndat*nant*nchan*npol*ndim

            self.info("ndat=" + str(ndat) + " bytes_per_sample=" +
                      str(bytes_per_sample) + " nval=" + str(nval))

            raw = np.fromfile(gains_file, dtype=np.float32, count=nval)
            gains_file.close()

            self.info("np.shape(raw)=" + str(np.shape(raw)))
            self.info("npol=" + str(npol) + " nchan=" + str(nchan))

            # reshape the raw data into a numpy array with specified dimensions
            data = raw.reshape((ndat, nant, npol, nchan, ndim))

            # acquire the results lock
            self.results["lock"].acquire()

            # generate an empty numpy array with ndat values
            xvals = np.zeros(ndat)
            gains_time = np.zeros((npol, ndat))
            gains_freq = np.zeros((npol, nchan))

            for idat in range(ndat):
                xvals[idat] = float(idat) * tsamp / 1e6
                for ipol in range(npol):
                    for isig in range(nant):
                        for ichan in range(nchan):
                            power = 0
                            for idim in range(ndim):
                                g = data[idat][isig][ipol][ichan][idim]
                                power += g * g
                            if power > gains_time[ipol][idat]:
                                gains_time[ipol][idat] = power
                            if idat == ndat-1:
                                gains_freq[ipol][ichan] = power

            if npol == 1:
                labels = ["AA+BB"]
                colours = ["red"]
            if npol == 2:
                labels = ["AA", "BB"]
                colours = ["red", "green"]
            else:
                labels = []
                colours = []

            self.gains_time_plot.plot(240, 180, True, xvals, gains_time,
                                      labels, colours)
            self.results["gainstime_lo"] = self.gains_time_plot.getRawImage()

            self.gains_time_plot.plot(1024, 768, False, xvals, gains_time,
                                      labels, colours)
            self.results["gainstime_hi"] = self.gains_time_plot.getRawImage()

            self.gains_freq_plot.plot_npol(240, 180, True, nchan, freq, bw,
                                           gains_freq, labels)
            self.results["gainsfreq_lo"] = self.gains_freq_plot.getRawImage()

            self.gains_freq_plot.plot_npol(1024, 768, False, nchan, freq, bw,
                                           gains_freq, labels)
            self.results["gainsfreq_hi"] = self.gains_freq_plot.getRawImage()

            self.gains_valid = True

            self.results["lock"].release()

            for file in files:
                os.rename(proc_dir + "/" + file, send_dir + "/" + file)
        return len(files)

    # process output from the calibration script
    def process_calibration(self, proc_dir, send_dir):

        files = [file for file in os.listdir(proc_dir)
                 if file.lower().endswith(".cal")]
        self.trace("files=" + str(files))

        for calfile in files:
            self.debug("processing " + proc_dir + "/" + calfile)
            atnf_calfile = self.atnf_utc(calfile)
            os.rename(proc_dir + "/" + calfile, send_dir + "/" + atnf_calfile)
        return len(files)


###############################################################################


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print "ERROR: 1 command line argument expected"
        sys.exit(1)

    # this should come from command line argument
    stream_id = sys.argv[1]

    script = []
    script = RepackPreprocessorDaemon("uwb_repack_preprocessor", stream_id)

    state = script.configure(DAEMONIZE, DL, "repack_preproc", "repack_preproc")

    if state != 0:
        sys.exit(state)

    script.log(1, "STARTING SCRIPT")

    try:

        reporting_thread = PreprocessorReportingThread(script, stream_id)
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
