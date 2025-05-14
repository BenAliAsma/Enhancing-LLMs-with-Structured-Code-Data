# Lines of code in types
This metric measures the number of lines of code for each type.

Large types can be problematic:

* They can be hard to understand and maintain, even with good tool support.
* They often arise as a result of bundling many unrelated things into the same type, and so can be a symptom of weak type cohesion.

## Recommendation
Types are generally too large because they are taking on more responsibilities than they should (see \[Martin\] for more on responsibilities). In general, the solution is to identify each of the different responsibilities the class is taking on, and split them out into multiple classes, e.g. using the 'Extract Class' refactoring from \[Fowler\].


## References
* M. Fowler. *Refactoring* pp. 65, 122-5. Addison-Wesley, 1999.
* R. Martin. *Agile Software Development: Principles, Patterns, and Practices* Chapter 8 - SRP: The Single-Responsibility Principle. Pearson Education, 2003.
