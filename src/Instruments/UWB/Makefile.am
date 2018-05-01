
noinst_LTLIBRARIES = libinstuwb.la

libinstuwb_headers = spip/UnpackFloatRAMUWB.h

libinstuwb_la_SOURCES = UnpackFloatRAMUWB.C

bin_PROGRAMS = uwb_adaptive_filter_pipeline \
  uwb_adaptive_filter_test \
  uwb_calibration_pipeline \
  uwb_continuum_pipeline \
  uwb_filterbank_pipeline

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

AM_CXXFLAGS += @PSRDADA_CFLAGS@ @HWLOC_CFLAGS@ @FFTW_CFLAGS@
LDADD += @PSRDADA_LIBS@ @HWLOC_LIBS@ @FFTW_LIBS@

uwb_adaptive_filter_test_SOURCES = uwb_adaptive_filter_test.C
uwb_adaptive_filter_pipeline_SOURCES = uwb_adaptive_filter_pipeline.C
uwb_continuum_pipeline_SOURCES = uwb_continuum_pipeline.C
uwb_calibration_pipeline_SOURCES = uwb_calibration_pipeline.C
uwb_filterbank_pipeline_SOURCES = uwb_filterbank_pipeline.C

if HAVE_CUDA

libinstuwb_la_SOURCES += UnpackFloatCUDAUWB.cu
libinstuwb_headers += spip/UnpackFloatCUDAUWB.h

AM_CXXFLAGS += @CUDA_CFLAGS@
LDADD += @CUDA_LIBS@

include $(top_srcdir)/config/Makefile.cuda

endif

