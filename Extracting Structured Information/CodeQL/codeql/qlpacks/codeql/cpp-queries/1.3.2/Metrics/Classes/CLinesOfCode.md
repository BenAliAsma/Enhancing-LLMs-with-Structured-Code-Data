# Lines of code per class
This metric measures the number of lines of code for each class.

Large classes can be problematic:

* They can be hard to understand and maintain, even with good tool support.
* They often arise as a result of bundling many unrelated things into the same class, and so can be a symptom of weak class cohesion.

## Recommendation
Classes are generally too large because they are taking on more responsibilities than they should (see \[Martin\] for more on responsibilities). In general, the solution is to identify each of the different responsibilities the class is taking on, and split them out into multiple classes, e.g. using the 'Extract Class' refactoring from \[Fowler\].


## References
* M. Fowler. *Refactoring* pp. 65, 122-5. Addison-Wesley, 1999.
* R. Martin. [The Single Responsibility Principle](https://drive.google.com/file/d/0ByOwmqah_nuGNHEtcU5OekdDMkk/view). Published online.
