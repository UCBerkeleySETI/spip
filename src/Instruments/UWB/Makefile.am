
noinst_LTLIBRARIES = libinstuwb.la

libinstuwb_headers = spip/UWBDigitizer.h

libinstuwb_la_SOURCES = UWBDigitizer.C

bin_PROGRAMS = uwb_digitizer

AM_CXXFLAGS = -I .\
  -I$(top_builddir)/src/Affinity \
  -I$(top_builddir)/src/Dada \
  -I$(top_builddir)/src/Util

LDADD = libinstuwb.la \
  $(top_builddir)/src/Dada/libspipdada.la \
  $(top_builddir)/src/Util/libspiputil.la \
  $(top_builddir)/src/Affinity/libspipaffinity.la \
  -lpthread

uwb_polsubxpose_SOURCES = uwb_digitizer.C

AM_CXXFLAGS += @PSRDADA_CFLAGS@ @HWLOC_CFLAGS@
LDADD += @PSRDADA_LIBS@ @HWLOC_LIBS@

if HAVE_CUDA

libinstuwb_headers += spip/UWB.h
libinstuwb_la_SOURCES += UWBDigitizer.C

AM_CXXFLAGS += @CUDA_CFLAGS@
LDADD += @CUDA_LIBS@

endif

