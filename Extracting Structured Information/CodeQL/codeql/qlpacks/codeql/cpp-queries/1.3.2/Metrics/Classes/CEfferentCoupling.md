# Outgoing dependencies per class
This metric measures the number of outgoing dependencies for each class, i.e. the number of other classes on which each class depends.

Classes that depend on many other classes are quite brittle, because if any of their dependencies change then they may have to as well. Furthermore, the reason for the high number of dependencies is often that different bits of the class depend on different sets of other classes, so it is not uncommon to find that classes with high efferent coupling also lack cohesion.


## Recommendation
Efferent coupling can be reduced by splitting a class into pieces along its dependency fault lines.


## References
* R. Martin. *Agile Software Development: Principles, Patterns and Practices*. Pearson, 2011.
* M. Fowler. *Refactoring*. Addison-Wesley, 1999.
* [Wikipedia: Code refactoring](https://en.wikipedia.org/wiki/Code_refactoring)
* [Refactoring as Meta Programming?](http://www.jot.fm/issues/issue_2005_01/column1/)
