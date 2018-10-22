
#ifndef __KeckRTCUtil_h
#define __KeckRTCUtil_h

#include <sys/time.h>
#include <vector>
#include <inttypes.h>

double diff_time ( struct timeval time1, struct timeval time2 );

void write_timing_data (const char * filename, std::vector<double>& frame_times,
                        uint64_t frames_sent);

void print_timing_data (std::vector<double>& frame_times, 
                        uint64_t frames_sent, uint64_t frame_size);


#endif
