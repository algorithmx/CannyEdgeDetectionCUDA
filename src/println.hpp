#pragma once

#include <iostream>

inline void println(const std::string &X) {
	std::cout << X << std::endl;
}

inline void errorln(const std::string &X){
	std::cerr << X << std::endl;
}

inline void errorln(const std::string &X, const std::string &Y){
	std::cerr << X << " " << Y << std::endl;
}