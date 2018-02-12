
noinst_LTLIBRARIES = libinstmeerkat.la

libinstmeerkat_headers = spip/MeerKATPolSubXpose.h

libinstmeerkat_la_SOURCES = MeerKATPolSubXpose.C

bin_PROGRAMS = meerkat_polsubxpose

AM_CXXFLAGS = -I .\
  -I$(top_builddir)/src/Affinity \
  -I$(top_builddir)/src/Dada \
  -I$(top_builddir)/src/Util

LDADD = libinstmeerkat.la \
  $(top_builddir)/src/Dada/libspipdada.la \
  $(top_builddir)/src/Util/libspiputil.la \
  $(top_builddir)/src/Affinity/libspipaffinity.la \
  -lpthread

meerkat_polsubxpose_SOURCES = meerkat_polsubxpose.C

AM_CXXFLAGS += @PSRDADA_CFLAGS@ @HWLOC_CFLAGS@
LDADD += @PSRDADA_LIBS@ @HWLOC_LIBS@

if HAVE_CUDA

libinstmeerkat_headers += spip/MeerKATPolSubXposeCUDA.h
libinstmeerkat_la_SOURCES += MeerKATPolSubXposeCUDA.C

AM_CXXFLAGS += @CUDA_CFLAGS@
LDADD += @CUDA_LIBS@

endif

