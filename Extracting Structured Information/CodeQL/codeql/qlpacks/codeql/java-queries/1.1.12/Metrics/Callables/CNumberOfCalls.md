# Number of calls in methods
If the number of calls that is made by a method (or constructor) to other methods is high, the method can be difficult to understand, because you have to read through all the methods that it calls to fully understand what it does. There are various reasons why a method may make a high number of calls, including:

* The method is simply too large in general.
* The method has too many responsibilities (see \[Martin\]).
* The method spends all of its time delegating rather than doing any work itself.

## Recommendation
The appropriate action depends on the reason why the method makes a high number of calls:

* If the method is too large, you should refactor it into multiple smaller methods, using the 'Extract Method' refactoring from \[Fowler\], for example.
* If the method is taking on too many responsibilities, a new layer of methods can be introduced below the top-level method, each of which can do some of the original work. The top-level method then only needs to delegate to a much smaller number of methods, which themselves delegate to the methods lower down.
* If the method spends all of its time delegating, some of the work that is done by the subsidiary methods can be moved into the top-level method, and the subsidiary methods can be removed. This is the refactoring called 'Inline Method' in \[Fowler\].

## References
* M. Fowler, *Refactoring*. Addison-Wesley, 1999.
* R. Martin. *Agile Software Development: Principles, Patterns, and Practices* Chapter 8 - SRP: The Single-Responsibility Principle. Pearson Education, 2003.
