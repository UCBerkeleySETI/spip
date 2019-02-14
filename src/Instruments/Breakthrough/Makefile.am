
noinst_LTLIBRARIES = libinstbl.la

libinstexamples_headers = spip/BlRequantClient.h

libinstexamples_la_SOURCES = BlRequantClient.C

bin_PROGRAMS = bl_requant_16b_to_8b

AM_CXXFLAGS = -I .\
  -I$(top_builddir)/src/Affinity \
  -I$(top_builddir)/src/Dada \
  -I$(top_builddir)/src/Util

LDADD = libinstexamples.la \
  $(top_builddir)/src/Dada/libspipdada.la \
  $(top_builddir)/src/Util/libspiputil.la \
  $(top_builddir)/src/Affinity/libspipaffinity.la \
  -lpthread

AM_CXXFLAGS += @PSRDADA_CFLAGS@ @HWLOC_CFLAGS@
LDADD += @PSRDADA_LIBS@ @HWLOC_LIBS@
