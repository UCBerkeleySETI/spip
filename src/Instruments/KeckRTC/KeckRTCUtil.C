
#include "spip/KeckRTCUtil.h"

double diff_time ( struct timeval time1, struct timeval time2 )
{
  return ( double(time2.tv_sec - time1.tv_sec) * 1000000 +
           double(time2.tv_usec - time1.tv_usec ) );
}

