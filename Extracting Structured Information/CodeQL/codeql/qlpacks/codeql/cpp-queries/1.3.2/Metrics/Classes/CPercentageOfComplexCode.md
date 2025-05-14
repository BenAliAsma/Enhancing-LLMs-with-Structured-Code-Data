# Percentage of complex code per class
This metric measures the percentage of the code within each class that is part of a complex function (defined to be a function that has a high cyclomatic complexity, i.e. there are a high number of linearly-independent execution paths through the function).

Functions with high cyclomatic complexity are typically difficult to understand and test. Classes whose code is primarily contained within such tricky functions are often strong candidates for refactoring.


## Recommendation
Each of the individual functions whose cyclomatic complexity is too high should be simplified, e.g. by tidying up complex logic and/or by splitting the function into multiple smaller functions using the 'Extract Method' refactoring from \[Fowler\]. If splitting the functions up results in a class with too many functions, the refactoring should be followed up with another one to resolve the new problem (as per the advice given for that situation).


## References

## References
* M. Fowler. *Refactoring*. Addison-Wesley, 1999.
* [Wikipedia: Code refactoring](https://en.wikipedia.org/wiki/Code_refactoring)
* [Refactoring as Meta Programming?](http://www.jot.fm/issues/issue_2005_01/column1/)
