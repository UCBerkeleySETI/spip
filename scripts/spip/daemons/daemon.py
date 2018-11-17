#!/usr/bin/env python

##############################################################################
#
#     Copyright(C) 2015 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
#
###############################################################################

from signal import signal, SIGINT
from atexit import register
import time
import errno
import os
import sys
import threading
import subprocess
import inspect

from spip.config import Config
from spip.utils import sockets
from spip.log_socket import LogSocket
from spip.threads.control_thread import ControlThread


class Daemon(object):

    def __init__(self, name, id):

        self.dl = 1

        self.name = name
        self.config = Config()
        self.cfg = self.config.getConfig()
        self.id = id
        self.hostname = sockets.getHostNameShort()

        self.req_host = ""
        self.beam_id = -1
        self.subband_id = -1

        self.control_thread = []
        self.log_sock = []
        self.binary_list = []

        self.log_dir = self.cfg["SERVER_LOG_DIR"]
        self.control_dir = self.cfg["SERVER_CONTROL_DIR"]

        # append the streamid/beamid/hostname
        self.name += "_" + str(self.id)

    def configure(self, become_daemon, dl, source, dest):

        # set the script debug level
        self.dl = dl

        # check the script is running on the configured host
        if self.req_host != self.hostname:
            sys.stderr.write("ERROR: script requires " +
                             self.req_host + ", but was launched on " +
                             self.hostname + "\n")
            return 1

        self.log_file = self.log_dir + "/" + dest + ".log"
        self.pid_file = self.control_dir + "/" + self.name + ".pid"
        self.quit_file = self.control_dir + "/" + self.name + ".quit"
        self.reload_file = self.control_dir + "/" + self.name + ".reload"

        # check if the script is running
        try:
            cmd = "pgrep -xfl 'python " + str(sys.argv[0]) + " " + \
                  str(self.id) + "'"
        except ValueError:
            cmd = "pgrep -xfl 'python " + str(sys.argv[0]) + "'"

        rval, lines = self.system(cmd, 3, quiet=True)
        if not rval and len(lines) > 1:
            sys.stderr.write("ERROR: script with same name exists launch " +
                             "pids=" + str(lines) + "\n")
            return 1

        # an existing PID file must be from a previous failed execution, delete
        if os.path.exists(self.pid_file):
            self.delpid

        if os.path.exists(self.quit_file):
            sys.stderr.write("ERROR: quit file existed at launch: " +
                             self.quit_file + "\n")
            return 1

        # optionally daemonize script
        if become_daemon:
            self.daemonize()

        # instansiate a threaded event signal
        self.quit_event = threading.Event()

        # install signal handler for SIGINT
        def signal_handler(sig, frame):
            sys.stderr.write("CTRL + C pressed\n")
            self.quit_event.set()

        signal(SIGINT, signal_handler)

        type = self.getBasis()
        self.configureLogs(source, dest, type)

        # constrain to cpu cores, if specified
        if not self.cpu_list == "-1":
            pid = str(os.getpid())
            cmd = "taskset -pc " + self.cpu_list + " " + pid
            self.debug(cmd)
            self.system(cmd, 2)

        # start a control thread to handle quit requests
        self.control_thread = ControlThread(self)
        self.control_thread.start()

        self.debug("log_file=" + self.log_file)

        self.debug("log_file=" + self.log_file)
        self.debug("pid_file=" + self.pid_file)
        self.debug("quit_file=" + self.quit_file)
        self.debug("reload_file=" + self.reload_file)

        return 0

    def daemonize(self):

        # standard input will always be directed to /dev/null
        stdin = "/dev/null"
        stdout = self.log_file
        stderr = self.log_file
        self.debug("log_file=" + self.log_file)

        try:
            pid = os.fork()
            if pid > 0:
                # exit first parent
                sys.exit(0)
        except OSError, e:
            sys.stderr.write("fork 1 failed: %d(%s)\n" % (e.errno, e.strerror))
            sys.exit(1)

        # decouple from parent environment
        os.chdir("/")
        os.setsid()
        os.umask(0)

        # do second fork
        try:
            pid = os.fork()
            if pid > 0:
                # exit from second parent
                sys.exit(0)
        except OSError, e:
            sys.stderr.write("fork 2 failed: %d(%s)\n" % (e.errno, e.strerror))
            sys.exit(1)

        # redirect standard file descriptors
        sys.stdout.flush()
        sys.stderr.flush()
        si = file(stdin, 'r')
        so = file(stdout, 'a+')
        se = file(stderr, 'a+', 0)
        os.dup2(si.fileno(), sys.stdin.fileno())
        os.dup2(so.fileno(), sys.stdout.fileno())
        os.dup2(se.fileno(), sys.stderr.fileno())

        # write pidfile, enable a function to cleanup pid file upon crash
        register(self.delpid)
        pid = str(os.getpid())
        file(self.pid_file, 'w+').write("%s\n" % pid)

    def delpid(self):
        if os.path.exists(self.pid_file):
            os.remove(self.pid_file)

    def configureLogs(self, source, dest, type):
        host = self.cfg["SERVER_HOST"]
        port = int(self.cfg["SERVER_LOG_PORT"])
        if self.log_sock:
            self.log_sock.close()
        self.log_sock = LogSocket(source, dest, self.id, type, host,
                                  port, self.dl)
        self.log_sock.connect(5)

    def caller_name(self, skip=2):
        """Get a name of a caller in the format module.class.method

           `skip` specifies how many levels of stack to skip while getting
           caller name. skip=1 means "who calls me", skip=2 "who calls my
           caller" etc.

           An empty string is returned if skipped levels exceed stack height
        """
        stack = inspect.stack()
        start = 0 + skip
        if len(stack) < start + 1:
            return ''
        parentframe = stack[start][0]

        name = []
        # module = inspect.getmodule(parentframe)
        # `modname` can be None when frame is executed directly in console
        # TODO(techtonik): consider using __main__
        # if module:
        #     name.append(module.__name__)
        # detect classname
        if 'self' in parentframe.f_locals:
            # I don't know any way to detect call from the object method
            # XXX: there seems to be no way to detect static method call
            #      - it will be just a function call
            name.append(parentframe.f_locals['self'].__class__.__name__)
        codename = parentframe.f_code.co_name
        if codename != '<module>':  # top level usually
            name.append(codename)  # function or a method

        # Avoid circular refs and frame leaks
        #  https://docs.python.org/2.7/library/inspect.html#the-interpreter-stack
        del parentframe, stack

        return ":".join(name)

    def info(self, message):
        caller = self.caller_name()
        self.log(1, message)

    def debug(self, message):
        caller = self.caller_name()
        self.log(2, caller + " " + message)

    def trace(self, message):
        caller = self.caller_name()
        self.log(3, caller + " " + message)

    def warn(self, message):
        caller = self.caller_name()
        self.log(-1, caller + " " + message)

    def error(self, message):
        caller = self.caller_name()
        self.log(-2, caller + " " + message)

    def log(self, level, message):
        if self.log_sock:
            if not self.log_sock.connected:
                self.log_sock.connect(1)
            self.log_sock.log(level, message)

    # check if any binaries exist
    def checkBinaries(self):

        self.debug("()")
        existed = False

        for binary in self.binary_list:
            cmd = "pgrep -f '^" + binary + "'"
            rval, lines = self.system(cmd, 3, quiet=True)
            self.debug("cmd=" + cmd + " rval=" + str(rval) + " lines=" +
                       str(lines))

            # if the binary exists, rval will be non zero
            existed = not rval

        return existed

    def tryKill(self, signal):

        self.debug("signal="+signal)

        existed = False

        # check each binary in the list
        for binary in self.binary_list:

            # check if the binary is running
            cmd = "pgrep -f '^" + binary + "'"
            rval, lines = self.system(cmd, 3, quiet=True)
            self.trace("Daemon::tryKill cmd=" + cmd + " rval=" + str(rval) +
                       " lines=" + str(lines))

            # if the binary exists, then kill with the specified signal
            if not rval:
                cmd = "pkill -SIG" + signal + " -f '^" + binary + "'"
                rval, lines = self.system(cmd, 2)

        time.sleep(1)

        for binary in self.binary_list:

            # check if the binary is still running
            cmd = "pgrep -f '^" + binary + "'"
            rval, lines = self.system(cmd, 3, quiet=True)
            self.debug("cmd=" + cmd + " rval=" + str(rval) +
                       " lines=" + str(lines))
            if not rval:
                existed = True

        return existed

    def killBinaries(self):

        signal_required = "None"

        self.debug("checkBinaries()")
        existed = self.checkBinaries()
        self.trace("checkBinaries() existed=" + str(existed))

        # if a binary is running
        if existed:
            signal_required = "INT"
            self.trace("tryKill(INT)")
            existed = self.tryKill("INT")
            self.trace("tryKill(INT) success=" + str(not existed))

            # if a binary is running after SIGINT
            if existed:
                time.sleep(1)
                signal_required = "TERM"
                self.trace("tryKill(TERM)")
                existed = self.tryKill("TERM")
                self.trace("tryKill(TERM) success=" + str(not existed))

                # if a binary is running after SIGTERM
                if existed:
                    time.sleep(1)
                    signal_required = "KILL"
                    self.trace("tryKill(KILL)")
                    existed = self.tryKill("KILL")
                    self.trace("tryKill(KILL) success=" + str(not existed))

        self.debug("signal_required=" + signal_required + " success=" +
                   str(not existed))

    def conclude(self):

        self.debug("")

        self.quit_event.set()

        self.debug("killBinaries")
        self.killBinaries()

        if self.control_thread:
            self.control_thread.join()

        if self.log_sock:
            self.log_sock.close()

    def system(self, command, dl=2, quiet=False, env_vars=os.environ.copy()):
        lines = []
        return_code = 0

        self.log(dl, "system: " + command)

        # setup the module object
        proc = subprocess.Popen(command,
                                env=env_vars,
                                shell=True,
                                stdin=None,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)

        # communicate the command
        try:
            (output, junk) = proc.communicate()
        except IOError, e:
            if e.errno == errno.EINTR:
                self.quit_event.set()
                return(-1, ("SIGINT"))

        return_code = proc.returncode

        if return_code and not quiet:
            self.log(0, "spip.system: " + command + " failed")

        # Once you have a valid response, split the return output
        if output:
            lines = output.rstrip('\n').split('\n')
            if dl <= self.dl or return_code:
                for line in lines:
                    self.log(0, "system: " + line)

        return return_code, lines

    def system_raw(self, command, dl=2, env_vars=os.environ.copy()):
        return_code = 0

        self.log(dl, "system: " + command)

        # setup the module object
        proc = subprocess.Popen(command,
                                env=env_vars,
                                shell=True,
                                stdin=None,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)

        # communicate the command
        try:
            (output, junk) = proc.communicate()
        except IOError, e:
            if e.errno == errno.EINTR:
                self.quit_event.set()
                return(-1, ("SIGINT"))

        return_code = proc.returncode

        if return_code:
            self.log(0, "spip.system: " + command + " failed")

        return return_code, output

    def system_piped(self, command, pipe, dl=2, env_vars=os.environ.copy()):

        return_code = 0

        if dl == 2:
            self.debug(command)

        # setup the module object
        proc = subprocess.Popen(command,
                                env=env_vars,
                                shell=True,
                                stdin=None,
                                stdout=pipe,
                                stderr=pipe)

        # now wait for the process to complete
        proc.wait()

        # discard the return code
        return_code = proc.returncode

        if return_code and not self.quit_event.isSet():
            self.log(0, "system_piped: " + command + " failed")

        return return_code
