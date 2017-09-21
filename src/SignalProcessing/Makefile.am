
noinst_LTLIBRARIES = libspipdsp.la

libspipdsp_headers = spip/Container.h \
	spip/ContainerRAM.h \
	spip/ContainerRing.h spip/ContainerRingRead.h spip/ContainerRingWrite.h \
	spip/DelayPipeline.h \
	spip/FractionalDelay.h \
	spip/IntegerDelay.h \
	spip/Transformation.h

libspipdsp_la_SOURCES = Container.C \
	ContainerRAM.C \
	ContainerRing.C ContainerRingRead.C ContainerRingWrite.C \
	DelayPipeline.C \
	FractionalDelay.C \
	Filterbank.C \
  BatchedBackwardFFT.C \
  ForwardFFT.C \
	IntegerDelay.C \
  UnpackFloat.C UnpackFloatRAM.C

bin_PROGRAMS = delay_pipeline filterbank_pipeline

delay_pipeline_SOURCES = delay_pipeline.C
filterbank_pipeline_SOURCES = filterbank_pipeline.C

AM_CXXFLAGS = @PSRDADA_CFLAGS@ \
	-I$(top_builddir)/src/Affinity\
	-I$(top_builddir)/src/Util \
	-I$(top_builddir)/src/Dada

LDADD = libspipdsp.la \
	$(top_builddir)/src/Affinity/libspipaffinity.la \
	$(top_builddir)/src/Dada/libspipdada.la \
	$(top_builddir)/src/Util/libspiputil.la \
	@PSRDADA_LIBS@

if HAVE_FFTW3

libspipdsp_la_SOURCES += ForwardFFTFFTW.C \
  BatchedBackwardFFTFFTW.C

AM_CXXFLAGS += @FFTW_CFLAGS@

LDADD += @FFTW_LIBS@

endif

if HAVE_CUFFT

libspipdsp_la_SOURCES += ForwardFFTCUDA.C \
  BatchedBackwardFFTCUDA.C

AM_CXXFLAGS += @CUFFT_CFLAGS@

LDADD += @CUFFT_LIBS@

endif

if HAVE_CUDA

libspipdsp_la_SOURCES += CUDAtoRAMTransfer.C \
  RAMtoCUDATransfer.C \
  ContainerCUDA.C \
  UnpackFloatCUDA.cu

AM_CXXFLAGS += @CUDA_CFLAGS@
LDADD += @CUDA_LIBS@

include $(top_srcdir)/config/Makefile.cuda

endif
