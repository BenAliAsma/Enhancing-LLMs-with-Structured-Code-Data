# Number of statements in methods
This metric measures the number of statements for each method.

Methods that consist of too many statements are hard to understand, difficult to check and a common source of bugs (particularly towards the end of the method, since few people ever read that far). They often lack cohesion because they are trying to do too many things at once, and should be refactored into multiple, smaller methods. As a rough guide, methods should be able to fit on a single screen or side of A4. Anything longer than that increases the risk of introducing new defects during routine code changes.


## Recommendation
Over-long methods should be broken up into smaller ones by extracting parts of their functionality out into auxiliary methods, using the technique that Martin Fowler's *Refactoring* book calls 'Extract Method' (see References).


## References
* M. Fowler. *Refactoring* pp. 89-95. Addison-Wesley, 1999.
