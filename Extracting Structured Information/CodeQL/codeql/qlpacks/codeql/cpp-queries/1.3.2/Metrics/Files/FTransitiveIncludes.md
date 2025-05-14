# Indirect includes per file
This metric measures the number of files that are directly or indirectly included using `#include`.

The value of this metric is usually directly correlated to the file's build time: the more included files, the longer the compilation time.

Often, files with a large number of includes do not require most of the included definitions, so they are contributing to unnecessarily long build times.


## Recommendation
* Remove redundant `#include` directives
* Use the specific header file required, if possible, rather than a high-level header that includes many other header files as well
* Split header files that contain lots of unrelated definitions or include large unrelated header files

## References
* [Header files](http://www.learncpp.com/cpp-tutorial/19-header-files/)
* [Decoupling C Header Files](http://www.drdobbs.com/cpp/decoupling-c-header-files/212701130)
* [C++ Best Practice - Designing Header Files](https://accu.org/journals/overload/14/72/griffiths_1995/)
