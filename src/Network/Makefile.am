
noinst_LTLIBRARIES = libspipnet.la

libspipnet_headers = spip/Socket.h spip/UDPSocket.h spip/UDPSocketReceive.h spip/UDPSocketSend.h \
                     spip/UDPGenerator.h spip/UDPReceiver.h spip/UDPStats.h spip/UDPOverflow.h \
                     spip/UDPFormat.h spip/TCPDummyServer.h spip/TCPSocket.h spip/TCPSocketServer.h

libspipnet_la_SOURCES = Socket.C C UDPSocket.C UDPSocketReceive.C UDPSocketSend.C \
                        UDPGenerator.C UDPReceiver.C UDPStats.C UDPOverflow.C \
                        UDPFormat.C TCPDummyServer.C TCPSocket.C TCPSocketServer.C

AM_CXXFLAGS = -I$(top_builddir)/src/Affinity \
							-I$(top_builddir)/src/Util \
  						-I/opt/cross/include -std=c++0x

LDADD = $(top_builddir)/src/Affinity/libspipaffinity.la \
			  $(top_builddir)/src/Util/libspiputil.la

if HAVE_PSRDADA

libspipnet_headers += spip/UDPReceiveDB.h spip/UDPReceiveDBStats.h spip/UDPReceiveMergeDB.h spip/UDPReceiveMerge2DB.h
 
libspipnet_la_SOURCES += UDPReceiveDB.C UDPReceiveDBStats.C UDPReceiveMergeDB.C UDPReceiveMerge2DB.C

AM_CXXFLAGS += @PSRDADA_CFLAGS@ -I$(top_builddir)/src/Dada
LDADD += $(top_builddir)/src/Dada/libspipdada.la

endif


if HAVE_SPEAD2

libspipnet_la_SOURCES += \
  SPEADReceiver.C \
  SPEADBeamFormerConfig.C \
  SPEADReceiverMerge.C

# libspipnet_headers += spip/IBVQueue.h

AM_CXXFLAGS += @BOOST_CPPFLAGS@ @SPEAD2_CFLAGS@
LDADD += @BOOST_SYSTEM_LDFLAGS@ @SPEAD2_LIBS@

if HAVE_PSRDADA

libspipnet_la_SOURCES += SPEADReceiveDB.C SPEADReceiveMergeDB.C

endif

endif

if HAVE_CUDA

AM_CXXFLAGS += @CUDA_CFLAGS@
LDADD += @CUDA_LIBS@

endif

if HAVE_VMA

libspipnet_la_SOURCES += UDPSocketReceiveVMA.C

libspipnet_headers += spip/UDPSocketReceiveVMA.h

endif
