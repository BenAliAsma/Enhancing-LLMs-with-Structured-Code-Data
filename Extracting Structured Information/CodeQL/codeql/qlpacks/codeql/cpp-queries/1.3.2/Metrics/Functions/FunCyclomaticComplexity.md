# Cyclomatic complexity of functions
This metric measures the cyclomatic complexity of each function in the project.

The cyclomatic complexity of a function is an indication of the number of paths that can be taken during the execution of a function. Straight-line code has zero cyclomatic complexity, while branches and loops increase cyclomatic complexity. A cyclomatic complexity above 50 should be considered bad practice and above 75 should definitely be addressed.

Functions with high cyclomatic complexity suffer from the following problems:

* Difficult to test since tests should be provided for each possible execution
* Difficult to understand since a developer needs to understand how all conditions interact
* Difficult to maintain since many execution paths is an indication of functions that perform too many tasks

## Recommendation
The primary way to reduce the complexity is to extract sub-functionality into separate functions. This improves on all problems described above. If the function naturally breaks up into a sequence of operations it is preferable to extract each operation as a separate function. Even if that's not the case it is often possible to extract the body of an iteration into a separate function to reduce complexity. If the complexity can't be reduced significantly make sure that the function is properly documented and carefully tested.


## References
* [Functions](http://www.cplusplus.com/doc/tutorial/functions/)
* M. Fowler. *Refactoring*. Addison-Wesley, 1999.
* [Wikipedia: Code refactoring](https://en.wikipedia.org/wiki/Code_refactoring)
* [Refactoring as Meta Programming?](http://www.jot.fm/issues/issue_2005_01/column1/)
