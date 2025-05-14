# Functions per file
This metric measures the number of functions defined in each file.

Tracking this metric over time will indicate which parts of the system are under active development. Cross-referencing with the other metrics "Cyclomatic Complexity" and "Lines of Code" is recommended, because files with high values for all three metrics are very likely to be too big and unwieldy; such files should be split up.


## Recommendation
If a file is too big, identify the different tasks that are carried out by its functions and split the file according to these tasks.


## References
* [Functions](http://www.cplusplus.com/doc/tutorial/functions/)
* M. Fowler. *Refactoring*. Addison-Wesley, 1999.
* [Wikipedia: Code refactoring](https://en.wikipedia.org/wiki/Code_refactoring)
* [Refactoring as Meta Programming?](http://www.jot.fm/issues/issue_2005_01/column1/)
