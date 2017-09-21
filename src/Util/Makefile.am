
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

TESTS = test_AsciiHeader

check_PROGRAMS = $(TESTS)

test_AsciiHeader_SOURCES = test_AsciiHeader.C

LDADD = libspiputil.la

