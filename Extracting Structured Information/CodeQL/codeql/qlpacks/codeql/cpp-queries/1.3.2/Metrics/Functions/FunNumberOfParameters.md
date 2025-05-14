# Number of parameters per function
This metric measures the number of formal parameters per function.

Functions with too many formal parameters are hard to use and maintain due to their complex interface. Using them is hard since you need to provide the right number of parameters in the right order. This can be particularly prone to mistakes if many of the parameters share the same type.


## Recommendation
If the function takes many related parameters they should be grouped into a class or struct. This improves the usability of the function with more explicit naming, and also avoids reduces the risk of forgetting parameters and accidental reordering. If some of the parameters are optional, either explicitly with default value, or accepting NULL or 0, then it may be better to create separate functions for the different use cases.


## References
* [Functions](http://www.cplusplus.com/doc/tutorial/functions/)
* M. Fowler. *Refactoring*. Addison-Wesley, 1999.
* [Wikipedia: Code refactoring](https://en.wikipedia.org/wiki/Code_refactoring)
* [Refactoring as Meta Programming?](http://www.jot.fm/issues/issue_2005_01/column1/)
