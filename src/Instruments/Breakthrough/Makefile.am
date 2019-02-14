
noinst_LTLIBRARIES = libinstbl.la

libinstbl_headers = \
  spip/BlRequant16ob8iClient.h \
  spip/BlRequant8i2uClient.h

libinstbl_la_SOURCES = \
  BlRequant16ob8iClient.C \
  BlRequant8i2uClient.C

bin_PROGRAMS = \
  bl_requant_16b_to_8b \
  bl_requant_8b_to_2b

AM_CXXFLAGS = -I .\
  -I$(top_builddir)/src/Affinity \
  -I$(top_builddir)/src/Dada \
  -I$(top_builddir)/src/Util

LDADD = libinstbl.la \
  $(top_builddir)/src/Dada/libspipdada.la \
  $(top_builddir)/src/Util/libspiputil.la \
  $(top_builddir)/src/Affinity/libspipaffinity.la \
  -lpthread

AM_CXXFLAGS += @PSRDADA_CFLAGS@ @HWLOC_CFLAGS@ @CUDA_CFLAGS@
LDADD += @PSRDADA_LIBS@ @HWLOC_LIBS@
