
noinst_LTLIBRARIES = libmeerkat.la

libmeerkat_headers = spip/UDPFormatMeerKATSimple.h spip/BlockFormatMeerKAT.h

libmeerkat_la_SOURCES = UDPFormatMeerKATSimple.C BlockFormatMeerKAT.C

bin_PROGRAMS = meerkat_udpgen meerkat_udprecv meerkat_dummyserver

meerkat_udpgen_SOURCES = meerkat_udpgen.C
meerkat_udprecv_SOURCES = meerkat_udprecv.C
meerkat_dummyserver_SOURCES = meerkat_dummyserver.C

AM_CXXFLAGS = -I. \
	-I$(top_builddir)/src/Network \
	-I$(top_builddir)/src/Affinity \
	-I$(top_builddir)/src/Util

LDADD = libmeerkat.la \
  $(top_builddir)/src/libspip.la

if HAVE_PSRDADA

bin_PROGRAMS += meerkat_udpdb meerkat_dbstats meerkat_simdb meerkat_udpmergedb meerkat_udpmerge2db

AM_CXXFLAGS += -I$(top_builddir)/src/Dada

meerkat_udpdb_SOURCES = meerkat_udpdb.C
meerkat_dbstats_SOURCES = meerkat_dbstats.C
meerkat_simdb_SOURCES = meerkat_simdb.C
meerkat_udpmergedb_SOURCES = meerkat_udpmergedb.C
meerkat_udpmerge2db_SOURCES = meerkat_udpmerge2db.C

endif

if HAVE_SPEAD2

libmeerkat_headers += spip/UDPFormatMeerKATSPEAD.h spip/UDPFormatMeerKATSPEAD1k.h spip/UDPFormatMeerKATSPEAD2k.h

libmeerkat_la_SOURCES += UDPFormatMeerKATSPEAD.C UDPFormatMeerKATSPEAD1k.C UDPFormatMeerKATSPEAD2k.C

bin_PROGRAMS += meerkat_speadrecv \
	meerkat_speadrecvmerge \
  meerkat_speadmeta \
	meerkat_ibvrecv \
  meerkat_ibvmerge2db \
  meerkat_ibvmerge2 \
	meerkat_speadtimestamp

meerkat_speadrecv_SOURCES = meerkat_speadrecv.C
meerkat_speadrecvmerge_SOURCES = meerkat_speadrecvmerge.C
meerkat_speadmeta_SOURCES = meerkat_speadmeta.C
meerkat_speadtimestamp_SOURCES = meerkat_speadtimestamp.C

if HAVE_PSRDADA 

bin_PROGRAMS += meerkat_speaddb \
	meerkat_ibvdb

meerkat_speaddb_SOURCES = meerkat_speaddb.C

endif

endif

AM_CXXFLAGS += @PSRDADA_CFLAGS@ @HWLOC_CFLAGS@ @BOOST_CPPFLAGS@ @SPEAD2_CFLAGS@ @CUDA_CFLAGS@

LDADD += @SPEAD2_LIBS@ @PSRDADA_LIBS@ @HWLOC_LIBS@ @CUDA_LIBS@
