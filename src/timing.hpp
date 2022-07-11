#pragma once

#include <iostream>
#include <boost/chrono.hpp>
#include <boost/thread/thread.hpp> 

namespace bch = boost::chrono ;

const auto time_unit_wait_for_acquiring_lock = bch::milliseconds(10) ;

inline void wait() {
#ifdef __DEBUG__
	// deadlock bug will cause infinite printing
	std::cout << "waiting ..." << std::endl;
#endif
	boost::this_thread::sleep_for(time_unit_wait_for_acquiring_lock); 
}

// usage:
// new_timing_point = timing(previous_time_point, message)
auto timing(bch::time_point<bch::high_resolution_clock> T0, const std::string & msg) {
	auto T1 = bch::high_resolution_clock::now();
	std::cout << msg << " : " << bch::duration_cast<bch::milliseconds>(T1-T0) << std::endl;
	return T1;
}
