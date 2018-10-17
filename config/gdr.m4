dnl @synopsis SWIN_LIB_GDR
dnl
AC_DEFUN([SWIN_LIB_GDR],
[
  AC_PROVIDE([SWIN_LIB_GDR])

  AC_REQUIRE([SWIN_PACKAGE_OPTIONS])
  SWIN_PACKAGE_OPTIONS([gdr])

  AC_MSG_CHECKING([for GDR library installation])

  if test x"$GDR" == x; then
    GDR=gdrapi
  fi

  if test "$have_gdr" != "user disabled"; then

    SWIN_PACKAGE_FIND([gdr],[gdrapi.h])
    SWIN_PACKAGE_TRY_COMPILE([gdr],[#include <gdrapi.h>])

    SWIN_PACKAGE_FIND([gdr],[lib$GDR.*])
    SWIN_PACKAGE_TRY_LINK([gdr],[#include <gdrapi.h>],
                      [gdr_t g],
                      [-l$GDR])

  fi

  AC_MSG_RESULT([$have_gdr])

  if test x"$have_gdr" = xyes; then

    AC_DEFINE([HAVE_GDR],[1],
              [Define if the GDR Library is present])
    [$1]

  else
    :
    [$2]
  fi

  GDR_LIBS="$gdr_LIBS"
  GDR_CFLAGS="$gdr_CFLAGS"

  AC_SUBST(GDR_LIBS)
  AC_SUBST(GDR_CFLAGS)
  AM_CONDITIONAL(HAVE_GDR,[test "$have_gdr" = yes])

])

