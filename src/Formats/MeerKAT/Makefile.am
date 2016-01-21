
lib_LTLIBRARIES = libmeerkat.la

libmeerkat_headers = spip/UDPFormatMeerKATSimple.h spip/BlockFormatMeerKAT.h

libmeerkat_la_SOURCES = UDPFormatMeerKATSimple.C BlockFormatMeerKAT.C

bin_PROGRAMS = meerkat_udpdb meerkat_udpgen meerkat_udprecv meerkat_dbstats

AM_CXXFLAGS = -I. \
	-I$(top_builddir)/src/Network \
	-I$(top_builddir)/src/Dada \
	-I$(top_builddir)/src/Affinity

LDADD = libmeerkat.la \
	$(top_builddir)/src/Network/libspipnet.la \
	$(top_builddir)/src/Dada/libspipdada.la \
	$(top_builddir)/src/Affinity/libspipaffinity.la \
	-L/opt/cross/lib

if HAVE_SPEAD2

libmeerkat_headers += spip/UDPFormatMeerKATSPEAD.h

libmeerkat_la_SOURCES += UDPFormatMeerKATSPEAD.C

bin_PROGRAMS += meerkat_speadrecv meerkat_speaddb

endif

AM_CXXFLAGS += @PSRDADA_CFLAGS@ @HWLOC_CFLAGS@ @SPEAD2_CFLAGS@
LDADD += @PSRDADA_LIBS@ @HWLOC_LIBS@ @SPEAD2_LIBS@