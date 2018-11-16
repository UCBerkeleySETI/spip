
noinst_LTLIBRARIES = libspipdsp.la

libspipdsp_headers = \
  spip/Add.h spip/AddTime.h spip/AddTimeRAM.h \
  spip/Append.h spip/AppendFrequency.h spip/AppendFrequnecyRAM.h spip/AppendTime.h spip/AppendTimeRAM.h\
  spip/CalibrationPipeline.h \
  spip/Combination.h \
  spip/ContinuumPipeline.h \
  spip/Container.h \
	spip/ContainerRAM.h \
	spip/ContainerRAMFileWrite.h \
	spip/ContainerRing.h spip/ContainerRingRead.h spip/ContainerRingWrite.h \
	spip/ContainerBufferedRing.h spip/ContainerBufferedRingWrite.h \
	spip/ContainerFile.h spip/ContainerFileWrite.h spip/ContainerFileRead.h \
	spip/Detection.h spip/DetectionPolarimetry.h spip/DetectionPolarimetryRAM.h \
	spip/DetectionSquareLaw.h spip/DetectionSquareLawRAM.h \
	spip/FAppendPipeline.h \
	spip/FractionalDelay.h \
  spip/File.h spip/FileWrite.h \
  spip/FoldTime.h \
  spip/HasInputs.h \
	spip/IntegerDelayRAM.h \
	spip/Integration.h spip/IntegrationRAM.h \
	spip/IntegrationBinned.h spip/IntegrationBinnedRAM.h \
  spip/PBSCalibrationPipeline.h \
  spip/PolCombine.h spip/PolCombineRAM.h \
  spip/PolSelect.h spip/PolSelectRAM.h \
	spip/RAMtoRAMTransfer.h \
	spip/ReverseFrequency.h spip/ReverseFrequencyRAM.h \
  spip/SampleFold.h spip/SampleFoldRAM.h\
	spip/Transformation.h \
  spip/AdaptiveFilterPipeline.h

libspipdsp_la_SOURCES = \
  Add.C AddTime.C AddTimeRAM.C \
  Append.C AppendFrequency.C AppendFrequencyRAM.C AppendTime.C AppendTimeRAM.C \
  CalibrationPipeline.C \
  ContinuumPipeline.C \
  Container.C \
	ContainerRAM.C \
	ContainerRAMFileWrite.C \
	ContainerRing.C ContainerRingRead.C ContainerRingWrite.C \
	ContainerBufferedRing.C ContainerBufferedRingWrite.C \
	ContainerFile.C ContainerFileWrite.C ContainerFileRead.C \
	Detection.C DetectionPolarimetry.C DetectionPolarimetryRAM.C \
	DetectionSquareLaw.C DetectionSquareLawRAM.C \
	FAppendPipeline.C \
	File.C FileWrite.C \
	Filterbank.C \
	FractionalDelay.C \
  FoldTime.C \
	Integration.C IntegrationRAM.C \
	IntegrationBinned.C IntegrationBinnedRAM.C \
	AdaptiveFilter.C AdaptiveFilterRAM.C \
  BatchedBackwardFFT.C \
  ForwardFFT.C \
  BackwardFFT.C \
  PBSCalibrationPipeline.C \
  PolCombine.C PolCombineRAM.C \
  PolSelect.C PolSelectRAM.C \
  RAMtoRAMTransfer.C \
  ReverseFrequency.C ReverseFrequencyRAM.C \
  SampleFold.C SampleFoldRAM.C \
  UnpackFloat.C UnpackFloatRAM.C \
  AdaptiveFilterPipeline.C  \
  AdaptiveFilterTest.C

AM_CXXFLAGS = @PSRDADA_CFLAGS@ \
	-I$(top_builddir)/src/Affinity\
	-I$(top_builddir)/src/Util \
	-I$(top_builddir)/src/Dada

libspipdsp_la_LIBADD = 

if HAVE_FFTW3

libspipdsp_la_SOURCES += ForwardFFTFFTW.C \
  BackwardFFTFFTW.C \
  BatchedBackwardFFTFFTW.C

AM_CXXFLAGS += @FFTW_CFLAGS@

libspipdsp_la_LIBADD += @FFTW_LIBS@

endif

if HAVE_CUFFT

libspipdsp_la_SOURCES += \
  ForwardFFTCUDA.cu \
  BackwardFFTCUDA.cu \
  BatchedBackwardFFTCUDA.C

libspipdsp_headers += \
  spip/ForwardFFTCUDA.h \
  spip/ackwardFFTCUDA.h \
  spip/BatchedBackwardFFTCUDA.h

AM_CXXFLAGS += @CUFFT_CFLAGS@

libspipdsp_la_LIBADD += @CUFFT_LIBS@

endif

if HAVE_CUDA

libspipdsp_headers += \
  spip/CUDAtoRAMTransfer.h \
  spip/RAMtoCUDATransfer.h \
  spip/ContainerCUDA.h spip/ContainerRingWriteCUDA.h spip/ContainerCUDAFileWrite.h \
  spip/DetectionPolarimetryCUDA.h \
  spip/DetectionSquareLawCUDA.h \
  spip/IntegrationCUDA.h \
  spip/IntegrationBinnedCUDA.h \
  spip/ReverseFrequencyCUDA.h \
  spip/PolSelectCUDA.h \
  spip/SampleFoldCUDA.h \
  spip/UnpackFloatCUDA.h \
  spip/AdaptiveFilterCUDA.h

libspipdsp_la_SOURCES += \
  CUDAtoRAMTransfer.C \
  RAMtoCUDATransfer.C \
  ContainerCUDA.C ContainerRingWriteCUDA.C ContainerCUDAFileWrite.C \
  DetectionPolarimetryCUDA.cu \
  DetectionSquareLawCUDA.cu \
  IntegrationCUDA.cu \
  IntegrationBinnedCUDA.cu \
  ReverseFrequencyCUDA.cu \
  PolSelectCUDA.cu \
  SampleFoldCUDA.cu \
  UnpackFloatCUDA.cu \
	AdaptiveFilterCUDA.cu

AM_CXXFLAGS += @CUDA_CFLAGS@
libspipdsp_la_LIBADD += @CUDA_LIBS@

include $(top_srcdir)/config/Makefile.cuda

endif
