
noinst_LTLIBRARIES = libinstkeckrtc.la

libinstkeckrtc_headers = \
  spip/KeckRTCDefs.h \
  spip/KeckRTCUtil.h

libinstkeckrtc_la_SOURCES = \
  KeckRTCUtil.C

bin_PROGRAMS = keckrtc_server keckrtc_client

AM_CXXFLAGS = -I .\
  -I$(top_builddir)/src/Affinity \
  -I$(top_builddir)/src/Network \
  -I$(top_builddir)/src/Util

LDADD = libinstkeckrtc.la \
  $(top_builddir)/src/Util/libspiputil.la \
  $(top_builddir)/src/Network/libspipnet.la \
  $(top_builddir)/src/Affinity/libspipaffinity.la \
  -lpthread

keckrtc_client_SOURCES = keckrtc_client.C
keckrtc_server_SOURCES = keckrtc_server.C

AM_CXXFLAGS += @HWLOC_CFLAGS@
LDADD += @HWLOC_LIBS@

if HAVE_CUDA

libinstkeckrtc_headers += spip/keckrtc_kernels.h
libinstkeckrtc_la_SOURCES += keckrtc_kernels.cu

AM_CXXFLAGS += @CUDA_CFLAGS@
LDADD += @CUDA_LIBS@

include $(top_srcdir)/config/Makefile.cuda

endif



