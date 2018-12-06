
noinst_LTLIBRARIES = libbpsr.la

libbpsr_headers = spip/UDPFormatBPSR.h spip/BlockFormatBPSR.h

libbpsr_la_SOURCES = UDPFormatBPSR.C BlockFormatBPSR.C

bin_PROGRAMS = bpsr_dbstats bpsr_udprecv bpsr_utctimestamp

bpsr_dbstats_SOURCES = bpsr_dbstats.C
bpsr_udprecv_SOURCES = bpsr_udprecv.C
bpsr_utctimestamp_SOURCES = bpsr_utctimestamp.C

AM_CXXFLAGS = -I. \
	-I$(top_builddir)/src/Affinity \
	-I$(top_builddir)/src/Network \
	-I$(top_builddir)/src/Util \
	-I$(top_builddir)/src/Dada

LDADD = libbpsr.la \
  $(top_builddir)/src/libspip.la

AM_CXXFLAGS += @PSRDADA_CFLAGS@ @HWLOC_CFLAGS@ @CUDA_CFLAGS@
LDADD += @PSRDADA_LIBS@ @HWLOC_LIBS@ @CUDA_LIBS@
