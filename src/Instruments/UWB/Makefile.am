
noinst_LTLIBRARIES = libinstuwb.la

libinstuwb_headers = \
  spip/PreprocessingPipeline.h \
  spip/UWBCalibrationPipeline.h \
  spip/UnpackFloatRAMUWB.h

libinstuwb_la_SOURCES = \
  PreprocessingPipeline.C \
  UWBCalibrationPipeline.C \
  UnpackFloatRAMUWB.C

bin_PROGRAMS = uwb_adaptive_filter_pipeline \
  uwb_adaptive_filter_combine \
  uwb_adaptive_filter_test \
  uwb_adaptive_filter_tappend \
  uwb_calibration_pipeline \
  uwb_continuum_pipeline \
  uwb_continuum_pipeline_float \
  uwb_continuum_scrunch \
  uwb_continuum_tadd \
  uwb_filterbank_pipeline \
  uwb_preprocessing_pipeline

AM_CXXFLAGS = -I .\
  -I$(top_builddir)/src/Affinity \
  -I$(top_builddir)/src/Dada \
  -I$(top_builddir)/src/SignalProcessing \
  -I$(top_builddir)/src/Util

LDADD = libinstuwb.la \
  $(top_builddir)/src/libspip.la

AM_CXXFLAGS += @PSRDADA_CFLAGS@ @HWLOC_CFLAGS@ @FFTW_CFLAGS@
LDADD += @PSRDADA_LIBS@ @HWLOC_LIBS@ @FFTW_LIBS@

uwb_adaptive_filter_combine_SOURCES = uwb_adaptive_filter_combine.C
uwb_adaptive_filter_test_SOURCES = uwb_adaptive_filter_test.C
uwb_adaptive_filter_tappend_SOURCES = uwb_adaptive_filter_tappend.C
uwb_adaptive_filter_pipeline_SOURCES = uwb_adaptive_filter_pipeline.C
uwb_continuum_pipeline_SOURCES = uwb_continuum_pipeline.C
uwb_continuum_pipeline_float_SOURCES = uwb_continuum_pipeline_float.C
uwb_calibration_pipeline_SOURCES = uwb_calibration_pipeline.C
uwb_filterbank_pipeline_SOURCES = uwb_filterbank_pipeline.C
uwb_preprocessing_pipeline_SOURCES = uwb_preprocessing_pipeline.C

if HAVE_CUDA

libinstuwb_la_SOURCES += UnpackFloatCUDAUWB.cu
libinstuwb_headers += spip/UnpackFloatCUDAUWB.h

AM_CXXFLAGS += @CUDA_CFLAGS@
LDADD += @CUDA_LIBS@

include $(top_srcdir)/config/Makefile.cuda

endif

