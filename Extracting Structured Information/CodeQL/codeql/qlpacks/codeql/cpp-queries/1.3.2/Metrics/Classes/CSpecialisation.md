# Specialization per class
This metric measures the extent to which a subclass overrides (replaces) the behavior of its ancestor classes. It is computed as follows: we determine the number of overridden functions in the subclass (not counting overrides of abstract functions), multiply by its depth in the inheritance hierarchy and then divide by its total number of functions.

If a class overrides many of the functions of its ancestor classes, it is an indication that the ancestor classes may not model sensible abstractions. This is particularly true for subclasses that are lower down in the inheritance hierarchy. In general, subclasses should add behavior to their superclasses, rather than redefining the behavior that is already there.


## Recommendation
The main reason that classes have a high specialization index is that multiple subclasses specialize a common base class in exactly the same way. In this case, the relevant function(s) should be pulled up into the base class (see the 'Pull Up Method' refactoring in \[Fowler\]).


## References
* M. Fowler. *Refactoring* pp. 260-3. Addison-Wesley, 1999.
* M. Lorenz and J. Kidd. *Object-oriented Software Metrics*. Prentice Hall, 1994.
* O. de Moor et al. *Keynote Address: .QL for Source Code Analysis*. Proceedings of the 7th IEEE International Working Conference on Source Code Analysis and Manipulation, 2007.
