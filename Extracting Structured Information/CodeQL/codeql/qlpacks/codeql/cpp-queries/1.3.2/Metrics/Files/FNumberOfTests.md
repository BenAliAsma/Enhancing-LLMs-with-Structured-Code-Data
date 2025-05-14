# Number of tests
This metric measures the number of tests in this file.

Tests are recognised if they are implemented with one of the major unit testing frameworks. Currently the Boost and CppUnit frameworks are recognised.

In general, having many test cases is a good thing rather than a bad thing. However, at the file level, tests should typically be grouped by the functionality they relate to, which makes a file with an exceptionally high number of tests a strong candidate for splitting up. At a higher level, this metric makes it possible to compare the number of tests in different components, potentially flagging functionality that is comparatively under-tested.


## Recommendation
Since it is typically not a problem to have too many tests, this metric is usually included for the purposes of collecting information, rather than finding problematic areas in the code. With that in mind, it is usually a good idea to avoid an excessive number of tests in a single file, and to maintain a broadly comparable level of testing across components.

When assessing the thoroughness of a code base's test suite, the number of test methods provides only part of the story. Test coverage statistics allow a more detailed examination of which parts of the code deserve improvements in this area.


## References
* Boost: official website at [http://www.boost.org/](http://www.boost.org/).
* CppUnit: official website at [https://freedesktop.org/wiki/Software/cppunit/](https://freedesktop.org/wiki/Software/cppunit/).
