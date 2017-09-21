
noinst_LTLIBRARIES = libspiputil.la

libspiputil_headers = spip/AsciiHeader.h \
	spip/BlockFormat.h \
	spip/Error.h \
	spip/Time.h

libspiputil_la_SOURCES = AsciiHeader.C  \
	BlockFormat.C \
	Error.C \
	tostring.C \
	Time.C

#if HAVE_CUFFT

libspiputil_headers += spip/CUFFTError.h

libspiputil_la_SOURCES += CUFFTError.C

#endif

AM_CPPFLAGS = @CUFFT_CFLAGS@

LDADD = @CUFFT_LIBS@
