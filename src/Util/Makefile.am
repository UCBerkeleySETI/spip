
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

libspiputil_la_LIBADD = 

#if HAVE_CUFFT

libspiputil_headers += spip/CUFFTError.h
libspiputil_la_SOURCES += CUFFTError.C
libspiputil_la_LIBADD += @CUFFT_LIBS@

AM_CPPFLAGS = @CUFFT_CFLAGS@
AM_LDFLAGS =  @CUFFT_LIBS@

#endif

TESTS = test_AsciiHeader

check_PROGRAMS = $(TESTS)

test_AsciiHeader_SOURCES = test_AsciiHeader.C

#test_AsciiHeader_LDADD = @CUFFT_LIBS@

