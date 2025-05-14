# Number of callables
This metric measures the number of methods per class. If a class has a large number of methods it is possible the class may have too many responsibilities which makes it more difficult to understand.


## Recommendation
If the class has too many responsibilities then some of them should be extracted into their own classes. If the class contains a lot of utility methods then consider moving them to another class too.

If the class is part of an inheritance hierarchy then consider whether the methods are in the right place. If the methods are present in multiple sibling classes or if the methods are only used in a small number of subclasses of the class in question then the hierarchy may need to be redesigned.


## References
* Martin Fowler. *Refactoring*. Addison-Wesley. 1999.
* MSDN. Microsoft Application Architecture Guide 2nd Edition. [Chapter 3: Architectural Patterns and Styles](http://msdn.microsoft.com/en-us/library/ee658117.aspx). Microsoft. 2009.
