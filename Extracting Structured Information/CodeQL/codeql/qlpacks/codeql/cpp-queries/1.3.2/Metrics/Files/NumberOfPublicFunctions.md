# Public functions per file
This metric measures the number of public functions in a file.

Public functions are the API between a component and other parts of a program. As such, it makes sense to monitor the number of public functions and consider the following questions:

* Is the API too large and unwieldy?
* When adding new functions, what effect will the new additions have on the cost of maintaining the additions (given that other components may come to rely on them)?
* Do any public functions inadvertently expose implementation details of a component?

## Recommendation
Adjust the API of a component in consideration of the above questions.


## References
* [Functions](http://www.cplusplus.com/doc/tutorial/functions/)
* M. Fowler. *Refactoring*. Addison-Wesley, 1999.
* [Wikipedia: Code refactoring](https://en.wikipedia.org/wiki/Code_refactoring)
* [Refactoring as Meta Programming?](http://www.jot.fm/issues/issue_2005_01/column1/)
