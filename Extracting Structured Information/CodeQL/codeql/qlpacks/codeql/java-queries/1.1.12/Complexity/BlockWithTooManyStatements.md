# Block with too many statements
Code has a tendency to become more complex over time. A method that is initially simple may need to be extended to accommodate additional functionality or to address defects. Before long it becomes unreadable and unmaintainable, with many complex statements nested within each other.

This rule applies to a block that contains a significant number of complex statements. Note that this is quite different from just considering the number of statements in a block, because each complex statement is potentially a candidate for being extracted to a new method as part of refactoring. For the purposes of this rule, loops and switch statements are considered to be complex.


## Recommendation
To make the code more understandable and less complex, identify logical units and extract them to new methods. As a result, the top-level logic becomes clearer.


## References
* M. Fowler, *Refactoring: Improving the Design of Existing Code*. Addison-Wesley Professional, 1999.
* W. C. Wake, *Refactoring Workbook*. Addison-Wesley Professional, 2004.
