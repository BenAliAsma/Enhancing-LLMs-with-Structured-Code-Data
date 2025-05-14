# Number of tests
This metric measures the number of tests below this location in the tree. At a file level, this would just be the number of test methods in the file.

A method is considered to be a "test method" if one of the major unit testing frameworks would invoke it as part of a test cycle. Recognized frameworks include NUnit, Microsoft Visual Studio Unit Testing Framework, and xUnit.net. For example, methods marked with the NUnit `Test` attribute are counted as test methods.

In general, having many test cases is a good thing rather than a bad thing. However, at the file level, tests should typically be grouped by the functionality they relate to, which makes a file with an exceptionally high number of tests a strong candidate for splitting up. At a higher level, this metric makes it possible to compare the number of tests in different components, potentially flagging functionality that is comparatively under-tested.


## Recommendation
Since it is typically not a problem to have too many tests, this metric is usually included for the purposes of collecting information, rather than finding problematic areas in the code. With that in mind, it is usually a good idea to avoid an excessive number of tests in a single file, and to maintain a broadly comparable level of testing across components.

When assessing the thoroughness of a code base's test suite, the number of test methods provides only part of the story. Test coverage statistics allow a more detailed examination of which parts of the code deserve improvements in this area.


## References
* NUnit: official website at [http://www.nunit.org/](http://www.nunit.org/).
* Microsoft Visual Studio Unit Testing Framework: documentation at [MSDN](https://msdn.microsoft.com/en-us/library/microsoft.visualstudio.testtools.unittesting.aspx).
* xUnit.net: official website at [https://xunit.net/](https://xunit.net/).
