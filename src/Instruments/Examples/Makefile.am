
noinst_LTLIBRARIES = libinstexamples.la

libinstexamples_headers = spip/ExampleReadWriteBlockClient.h

libinstexamples_la_SOURCES = ExampleReadWriteBlockClient.C

bin_PROGRAMS = example_readwriteblockclient

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

