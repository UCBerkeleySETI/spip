
noinst_LTLIBRARIES = libuwb.la

libuwb_headers = spip/BlockFormatUWB.h spip/UDPFormatUWB.h

libuwb_la_SOURCES = BlockFormatUWB.C UDPFormatUWB.C

bin_PROGRAMS = uwb_dbstats uwb_udprecv uwb_udpdb uwb_udpdbstats uwb_simdb

uwb_dbstats_SOURCES = uwb_dbstats.C
uwb_udprecv_SOURCES = uwb_udprecv.C
uwb_udpdb_SOURCES = uwb_udpdb.C
uwb_udpdbstats_SOURCES = uwb_udpdbstats.C
uwb_simdb_SOURCES = uwb_simdb.C

AM_CXXFLAGS = -I. \
	-I$(top_builddir)/src/Affinity \
	-I$(top_builddir)/src/Network \
	-I$(top_builddir)/src/Util \
	-I$(top_builddir)/src/Dada \
	-I$(top_builddir)/src/Formats/VDIF

LDADD = libuwb.la \
	$(top_builddir)/src/Network/libspipnet.la \
	$(top_builddir)/src/Dada/libspipdada.la \
	$(top_builddir)/src/Util/libspiputil.la \
	$(top_builddir)/src/Formats/VDIF/libvdif.la \
	$(top_builddir)/src/libspip.la -lpthread 

AM_CXXFLAGS += @PSRDADA_CFLAGS@ @HWLOC_CFLAGS@ @VDIFIO_CFLAGS@ @CUDA_CFLAGS@ @FFTW_CFLAGS@
LDADD += @PSRDADA_LIBS@ @HWLOC_LIBS@ @VDIFIO_LIBS@ @BOOST_SYSTEM_LDFLAGS@ @FFTW_LIBS@

if HAVE_SPEAD2

bin_PROGRAMS += uwb_ibvrecv
uwb_ibvrecv_SOURCES = uwb_ibvrecv.C

AM_CXXFLAGS += @SPEAD2_CFLAGS@
LDADD += @SPEAD2_LIBS@
uwb_ibvrecv_LDADD = $(LDADD) -lrdmacm -libverbs

endif

