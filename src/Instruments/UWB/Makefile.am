
noinst_LTLIBRARIES = libinstuwb.la

libinstuwb_headers = spip/ContinuumPipeline.h

libinstuwb_la_SOURCES = ContinuumPipeline.C

bin_PROGRAMS = uwb_continuum_pipeline

AM_CXXFLAGS = -I .\
  -I$(top_builddir)/src/Affinity \
  -I$(top_builddir)/src/Dada \
  -I$(top_builddir)/src/SignalProcessing \
  -I$(top_builddir)/src/Util

LDADD = libinstuwb.la \
  $(top_builddir)/src/SignalProcessing/libspipdsp.la \
  $(top_builddir)/src/Dada/libspipdada.la \
  $(top_builddir)/src/Util/libspiputil.la \
  $(top_builddir)/src/Affinity/libspipaffinity.la \
  -lpthread

AM_CXXFLAGS += @PSRDADA_CFLAGS@ @HWLOC_CFLAGS@
LDADD += @PSRDADA_LIBS@ @HWLOC_LIBS@

uwb_continuum_pipeline_SOURCES = uwb_continuum_pipeline.C

if HAVE_CUDA

AM_CXXFLAGS += @CUDA_CFLAGS@
LDADD += @CUDA_LIBS@

endif

