
SUBDIRS = Affinity Dada Network Util SignalProcessing . Formats Telescope Instruments

lib_LTLIBRARIES = libspip.la

libspip_la_SOURCES = spip.C

libspip_la_headers = spip.h

libspip_la_LIBADD = \
	Affinity/libspipaffinity.la \
  Dada/libspipdada.la \
	Network/libspipnet.la \
	Util/libspiputil.la \
	SignalProcessing/libspipdsp.la

