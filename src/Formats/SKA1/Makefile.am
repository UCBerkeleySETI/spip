
noinst_LTLIBRARIES = libska1.la

libska1_headers = spip/UDPFormatCustom.h

libska1_la_SOURCES = UDPFormatCustom.C

bin_PROGRAMS = ska1_udpgen ska1_udprecv

ska1_udpgen_SOURCES = ska1_udpgen.C
ska1_udprecv_SOURCES = ska1_udprecv.C

AM_CXXFLAGS = -I. \
  -I$(top_builddir)/src/Affinity \
	-I$(top_builddir)/src/Network \
	-I$(top_builddir)/src/Util

LDADD = libska1.la \
  $(top_builddir)/src/libspip.la

if HAVE_PSRDADA

bin_PROGRAMS += ska1_udpdb
AM_CXXFLAGS += -I$(top_builddir)/src/Dada

ska1_udpdb_SOURCES = ska1_udpdb.C

endif

AM_CXXFLAGS += @PSRDADA_CFLAGS@ @HWLOC_CFLAGS@ @CUDA_CFLAGS@
LDADD += @PSRDADA_LIBS@ @HWLOC_LIBS@ @CUDA_LIBS@

