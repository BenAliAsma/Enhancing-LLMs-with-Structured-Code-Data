# Outgoing dependencies
This metric calculates the number of outgoing dependencies of a type (the number of other types that it depends on).

Types that have a large number of outgoing type dependencies are very brittle as they may have to be changed if any of their dependencies are changed. It is not uncommon that a class with a high number of outgoing dependencies lacks cohesion because small parts of it rely on small parts of other classes.


## Recommendation
It might be possible to improve the class by splitting it into multiple classes. Methods that have disparate dependencies could be put in different classes.


## References
* Robert Cecil Martin. *Agile Software Development: Principles, Patterns and Practices*. Pearson Education. 2002.
