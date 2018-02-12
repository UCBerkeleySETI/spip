
noinst_LTLIBRARIES = libspipdada.la

libspipdada_headers = 
libspipdada_la_SOURCES = 
libspipdada_la_LIBADD = 
AM_CXXFLAGS =

if HAVE_PSRDADA

libspipdada_headers += \
	spip/DataBlock.h \
	spip/DataBlockRead.h \
	spip/DataBlockWrite.h \
	spip/DataBlockView.h \
	spip/DataBlockStats.h \
	spip/DadaClient.h \
	spip/ReadClient.h \
	spip/ReadReadWriteBlockClient.h \
	spip/ReadReadWriteClient.h \
	spip/ReadWriteBlockClient.h \
	spip/ReadWriteClient.h \
	spip/WriteClient.h \
	spip/WriteBlockClient.h \
	spip/SimReceiveDB.h

libspipdada_la_SOURCES += \
	DataBlock.C \
 	DataBlockRead.C \
	DataBlockWrite.C \
	DataBlockView.C \
	DataBlockStats.C \
  DadaClient.C \
  ReadClient.C \
  ReadReadWriteBlockClient.C \
  ReadReadWriteClient.C \
  ReadWriteBlockClient.C \
  ReadWriteClient.C \
  WriteClient.C \
  WriteBlockClient.C \
  SimReceiveDB.C

AM_CXXFLAGS += @PSRDADA_CFLAGS@
libspipdada_la_LIBADD += @PSRDADA_LIBS@

if HAVE_CUDA

libspipdada_headers += spip/CudaClient.h
libspipdada_la_SOURCES += CudaClient.C
AM_CXXFLAGS += @CUDA_CFLAGS@
libspipdada_la_LIBADD += @CUDA_LIBS@ 

endif
 
endif

AM_CXXFLAGS += \
	-I$(top_builddir)/src/Network \
	-I$(top_builddir)/src/Util

LDADD = \
  $(top_srcdir)/src/Network/libspipnet.la \
  $(top_srcdir)/src/Util/libspiputil.la

