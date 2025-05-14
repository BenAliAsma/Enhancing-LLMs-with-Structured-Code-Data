# Number of fields per class
This metric measures the number of fields in classes below this location in the tree. At a class level, it just measures the number of fields in the class itself.

There are at least a couple of reasons why a class may have too many fields:

* The class in general may be too big / have too many responsibilities.
* Several of the fields may be part of the same abstraction.

## Recommendation
The best resolution depends on the underlying reason behind it:

* If the class is too big, it should be split into multiple smaller classes.
* If several of the fields are part of the same abstraction, they should be grouped into a separate class, using the 'Extract Class' refactoring described in \[Fowler\].

## References
* M. Fowler. *Refactoring* pp. 65, 122-5. Addison-Wesley, 1999.
* R. Martin. [The Single Responsibility Principle](https://drive.google.com/file/d/0ByOwmqah_nuGNHEtcU5OekdDMkk/view). Published online.
