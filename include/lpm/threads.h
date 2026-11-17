// OpenMP threads that wait at a barrier spin by default (libgomp), and a spinning thread slows the
// thread that shares its core (SMT): with 32 threads on 16 cores the pipeline ran twice as slow as
// with passive waiting, and slower than with 16 threads. libgomp reads OMP_WAIT_POLICY when it is
// loaded, before main, so a program calls passive_omp_wait first thing, which executes the
// program again with OMP_WAIT_POLICY=passive unless the variable is already set.
#pragma once
#include <cstdlib>

#ifdef __linux__
#include <unistd.h>
#endif

namespace lpm {

inline void passive_omp_wait(char** argv) {
#ifdef __linux__
    if (std::getenv("OMP_WAIT_POLICY")) return;
    setenv("OMP_WAIT_POLICY", "passive", 0);
    execv("/proc/self/exe", argv);  // returns only on failure, and the program goes on as it is
#else
    (void)argv;
#endif
}

}  // namespace lpm
