
noinst_LTLIBRARIES = libspiptelescope.la

libspiptelescope_headers = spip/Antenna.h \
	spip/AntennaMolonglo.h \
	spip/Channel.h \
	spip/Delays.h

libspiptelescope_la_SOURCES = Antenna.C \
	AntennaMolonglo.C \
  Channel.C \
  Delays.C


AM_CXXFLAGS = \
  -I$(top_builddir)/src/Affinity \
  -I$(top_builddir)/src/SignalProcessing \ 
  -I$(top_builddir)/src/Util

LDADD = \
  $(top_builddir)/src/Affinity/libspipaffinity.la \
  $(top_builddir)/src/SignalProcessing/libspipdsp.la \
  $(top_builddir)/src/Util/libspiputil.la


