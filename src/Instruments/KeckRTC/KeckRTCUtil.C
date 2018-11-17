
#include "spip/KeckRTCUtil.h"

#include <unistd.h>
#include <fcntl.h>
#include <cfloat>
#include <iostream>

double diff_time ( struct timeval time1, struct timeval time2 )
{
  return ( double(time2.tv_sec - time1.tv_sec) * 1000000 +
           double(time2.tv_usec - time1.tv_usec ) );
}

void write_timing_data (const char * filename, std::vector<double>& frame_times, 
		        uint64_t frames_offset, uint64_t frames_sent)
{
  // write to the output file
  int flags = O_WRONLY | O_CREAT | O_TRUNC;
  int perms = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;
  int fd = open (filename, flags, perms);

  uint64_t frames_counted = frames_sent - frames_offset;
  ::write (fd, &frame_times[frames_offset], frames_counted * sizeof(double));
  ::close (fd);
}

void print_timing_data (std::vector<double>& frame_times, 
                        uint64_t frames_offset, uint64_t frames_sent, 
                        uint64_t frame_size)
{
  double duration_min = DBL_MAX;
  double duration_max = -DBL_MAX;
  double duration_sum = 0;

  double time_sum = 0;

  for (uint64_t iframe=frames_offset; iframe<frames_sent; iframe++)
  {
    double duration = frame_times[iframe];

    if (duration < duration_min)
      duration_min = duration;
    if (duration > duration_max)
      duration_max = duration;
    duration_sum += duration;

    time_sum += duration;
  }

  double frames_counted = double(frames_sent - frames_offset);

  double duration_mean = duration_sum / frames_counted;
  std::cerr << "duration timing:" << std::endl;
  std::cerr << "  minimum=" << duration_min << " us" << std::endl;
  std::cerr << "  mean="    << duration_mean << " us" << std::endl;
  std::cerr << "  maximum=" << duration_max << " us" << std::endl;

  double bytes_per_microsecond = double (frames_counted * frame_size) / time_sum;
  double gb_per_second = (bytes_per_microsecond * 1000000) / 1000000000;

  double frames_per_microsecond = frames_counted / time_sum;
  double frames_per_second = frames_per_microsecond *1e6;

  std::cerr << "  data_rate=" << gb_per_second << " Gb/s" << std::endl;
  std::cerr << "  frame_rate=" << frames_per_second << std::endl;
  std::cerr << "  frames sent=" << frames_sent << " counted=" << frames_counted << std::endl;
}

