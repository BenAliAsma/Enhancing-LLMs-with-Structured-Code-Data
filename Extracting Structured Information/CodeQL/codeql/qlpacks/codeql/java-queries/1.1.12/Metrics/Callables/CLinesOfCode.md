# Lines of code in methods
A method that contains a high number of lines of code has a number of problems:

* It can be difficult to understand, difficult to check, and a common source of defects (particularly towards the end of the method, because few people read that far).
* It is likely to lack cohesion because it has too many responsibilities.
* It increases the risk of introducing new defects during routine code changes.

## Recommendation
Break up long methods into smaller methods by extracting parts of their functionality into simpler methods, for example by using the 'Extract Method' refactoring from \[Fowler\]. As an approximate guide, a method should fit on one screen or side of Letter/A4 paper.


## References
* M. Fowler, *Refactoring*, pp. 89-95. Addison-Wesley, 1999.
