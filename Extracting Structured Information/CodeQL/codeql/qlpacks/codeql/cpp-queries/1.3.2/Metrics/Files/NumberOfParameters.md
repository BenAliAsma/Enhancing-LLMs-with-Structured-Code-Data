# Parameters per function
This metric measures the average number of parameters to functions.

Functions with an excessive number of parameters often exhibit one or more of the following problems.

* The functions lack cohesion because they do several unrelated tasks.
* Several of the arguments would be better grouped together using a struct or class, because they represent one conceptual entity.
* Calling the function is prone to mistakes because arguments may be accidentally transposed, possibly without a type-error.

## Recommendation
Consider refactoring functions with too many arguments into smaller functions, each with a single well-defined purpose.

It may also be possible to create new abstractions for usefully grouping arguments. For example, rather than representing a buffer by a pointer and an integer length, they could be encapsulated into a single struct, with utility functions for common operations.


## References
* M. Fowler. *Refactoring*. Addison-Wesley, 1999.
* [Wikipedia: Code refactoring](https://en.wikipedia.org/wiki/Code_refactoring)
* [Refactoring as Meta Programming?](http://www.jot.fm/issues/issue_2005_01/column1/)
