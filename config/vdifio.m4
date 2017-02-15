dnl @synopsis SWIN_LIB_VDIFIO
dnl
AC_DEFUN([SWIN_LIB_VDIFIO],
[
  AC_PROVIDE([SWIN_LIB_VDIFIO])

  AC_REQUIRE([SWIN_PACKAGE_OPTIONS])
  SWIN_PACKAGE_OPTIONS([vdifio])

  AC_MSG_CHECKING([for VDIFIO Library installation])

  if test x"$VDIFIO" == x; then
    VDIFIO=vdifio
  fi

  if test "$have_vdifio" != "user disabled"; then

    ac_tmp_CXXFLAGS="$CXXFLAGS"
    CXXFLAGS="$ac_tmp_CXXFLAGS -std=c++11"

    SWIN_PACKAGE_FIND([vdifio],[vdifio.h])
    SWIN_PACKAGE_TRY_COMPILE([vdifio],[#include <vdifio.h>])

    SWIN_PACKAGE_FIND([vdifio],[lib$VDIFIO.*])
    SWIN_PACKAGE_TRY_LINK([vdifio],[#include <vdifio.h>],
                      [vdif_header header],
                      [-l$VDIFIO])

  fi

  AC_MSG_RESULT([$have_vdifio])

  if test x"$have_vdifio" = xyes; then

    AC_DEFINE([HAVE_VDIFIO],[1],
              [Define if the VDIFIO Library is present])
    [$1]

  else
    :
    [$2]
  fi

  VDIFIO_LIBS="$vdifio_LIBS"
  VDIFIO_CFLAGS="$vdifio_CFLAGS"

  AC_SUBST(VDIFIO_LIBS)
  AC_SUBST(VDIFIO_CFLAGS)
  AM_CONDITIONAL(HAVE_VDIFIO,[test "$have_vdifio" = yes])

])

