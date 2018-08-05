
noinst_LTLIBRARIES = libspipdsp.la

libspipdsp_headers = spip/Container.h \
	spip/ContainerRAM.h \
	spip/ContainerRing.h spip/ContainerRingRead.h spip/ContainerRingWrite.h \
	spip/ContainerBufferedRing.h spip/ContainerBufferedRingWrite.h \
	spip/Detection.h spip/DetectionPolarimetry.h spip/DetectionPolarimetryRAM.h \
	spip/DetectionSquareLaw.h spip/DetectionSquareLawRAM.h \
	spip/FractionalDelay.h \
  spip/FoldTime.h \
	spip/IntegerDelayRAM.h \
	spip/Integration.h spip/IntegrationRAM.h \
  spip/SampleFold.h spip/SampleFoldRAM.h\
	spip/Transformation.h

libspipdsp_la_SOURCES = Container.C \
	ContainerRAM.C \
	ContainerRing.C ContainerRingRead.C ContainerRingWrite.C \
	ContainerBufferedRing.C ContainerBufferedRingWrite.C \
	Detection.C DetectionPolarimetry.C DetectionPolarimetryRAM.C \
	DetectionSquareLaw.C DetectionSquareLawRAM.C \
	Filterbank.C \
	FractionalDelay.C \
  FoldTime.C \
	Integration.C IntegrationRAM.C \
	AdaptiveFilter.C \
	AdaptiveFilterRAM.C \
	AdaptiveFilterPipeline.C \
	AdaptiveFilterTest.C \
  BatchedBackwardFFT.C \
  ForwardFFT.C \
  BackwardFFT.C \
  SampleFold.C SampleFoldRAM.C \
  UnpackFloat.C UnpackFloatRAM.C

bin_PROGRAMS = filterbank_pipeline adaptive_filter_pipeline adaptive_filter_test

filterbank_pipeline_SOURCES = filterbank_pipeline.C
adaptive_filter_pipeline_SOURCES = adaptive_filter_pipeline.C
adaptive_filter_test_SOURCES = adaptive_filter_test.C

AM_CXXFLAGS = @PSRDADA_CFLAGS@ \
	-I$(top_builddir)/src/Affinity\
	-I$(top_builddir)/src/Util \
	-I$(top_builddir)/src/Dada

LDADD = libspipdsp.la \
	$(top_builddir)/src/Affinity/libspipaffinity.la \
	$(top_builddir)/src/Dada/libspipdada.la \
	$(top_builddir)/src/Util/libspiputil.la \
	@PSRDADA_LIBS@

libspipdsp_la_LIBADD = @PSRDADA_LIBS@

if HAVE_FFTW3

libspipdsp_la_SOURCES += ForwardFFTFFTW.C \
  BackwardFFTFFTW.C \
  BatchedBackwardFFTFFTW.C

AM_CXXFLAGS += @FFTW_CFLAGS@

libspipdsp_la_LIBADD += @FFTW_LIBS@
LDADD += @FFTW_LIBS@

endif

if HAVE_CUFFT

libspipdsp_la_SOURCES += ForwardFFTCUDA.cu \
  BackwardFFTCUDA.cu \
  BatchedBackwardFFTCUDA.C

AM_CXXFLAGS += @CUFFT_CFLAGS@

libspipdsp_la_LIBADD += @CUFFT_LIBS@
LDADD += @CUFFT_LIBS@

endif

if HAVE_CUDA

libspipdsp_headers += spip/DetectionPolarimetryCUDA.h \
  DetectionSquareLawCUDA.h

libspipdsp_la_SOURCES += CUDAtoRAMTransfer.C \
  RAMtoCUDATransfer.C \
  ContainerCUDA.C \
  DetectionPolarimetryCUDA.cu \
  DetectionSquareLawCUDA.cu \
  IntegrationCUDA.cu \
  SampleFoldCUDA.cu \
  UnpackFloatCUDA.cu \
	AdaptiveFilterCUDA.cu

AM_CXXFLAGS += @CUDA_CFLAGS@
libspipdsp_la_LIBADD += @CUDA_LIBS@
LDADD += @CUDA_LIBS@

include $(top_srcdir)/config/Makefile.cuda

endif
