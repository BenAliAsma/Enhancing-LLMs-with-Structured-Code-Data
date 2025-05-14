# Number of statements per function
This metric measures the number of statements (including control statements such as `if` and `while`) in each function.

Functions with many statements are hard to understand and maintain. It is usually a sign of a function with too many responsibilities.


## Recommendation
The primary way to improve the complexity is to extract sub-functionality into separate functions. If the function naturally breaks up into a sequence of operations it is preferable to extract each operation as a separate function. This is most likely the case for large functions with low cyclomatic complexity. Even if the code is straight forward it is better to extract functionality for readability purposes. Moreover, it may enable reuse of the extracted subfunctionality.


## References
* [Functions](http://www.cplusplus.com/doc/tutorial/functions/)
* M. Fowler. *Refactoring*. Addison-Wesley, 1999.
* [Wikipedia: Code refactoring](https://en.wikipedia.org/wiki/Code_refactoring)
* [Refactoring as Meta Programming?](http://www.jot.fm/issues/issue_2005_01/column1/)
