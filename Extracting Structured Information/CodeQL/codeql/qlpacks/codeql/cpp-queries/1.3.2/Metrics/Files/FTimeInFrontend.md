# Compilation time
This metric measures the amount of time (in milliseconds) spent compiling a C/C++ file, including time spent processing all files included by the pre-processor.

Files that take too long to build usually have too many includes.


## Recommendation
Files that take a long time to build should be checked to see if they are including only the necessary files in order to reduce build time.


## References
* MSDN Library: [\#include directive (C/C++)](https://docs.microsoft.com/en-us/cpp/preprocessor/hash-include-directive-c-cpp)
* [Include operation](http://gcc.gnu.org/onlinedocs/cpp/Include-Operation.html#Include-Operation)
* [C++ Compilation Speed](http://www.drdobbs.com/cpp/c-compilation-speed/228701711)
