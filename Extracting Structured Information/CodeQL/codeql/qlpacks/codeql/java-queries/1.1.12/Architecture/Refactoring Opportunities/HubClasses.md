# Hub classes
A *hub class* is a class that depends on many other classes, and on which many other classes depend.

For the purposes of this rule, a *dependency* is any use of one class in another. Examples include:

* Using another class as the declared type of a variable or field
* Using another class as an argument type for a method
* Using another class as a superclass in the `extends` declaration
* Calling a method defined in the class
A class can be regarded as a hub class when both the incoming dependencies and the outgoing source dependencies are particularly high. (Outgoing source dependencies are dependencies on other source classes, rather than library classes like `java.lang.Object`.)

It is undesirable to have many hub classes because they are extremely difficult to maintain. This is because many other classes depend on a hub class, and so the other classes have to be tested and possibly adapted after each change to the hub class. Also, when one of a hub class's direct dependencies changes, the behavior of the hub class and all of its dependencies has to be checked and possibly adapted.


## Recommendation
One common reason for a class to be regarded as a hub class is that it tries to do too much, including unrelated functionality that depends on different parts of the code base. If possible, split such classes into several better encapsulated classes.

Another common reason is that the class is a "struct-like" class that has many fields of different types. Introducing some intermediate grouping containers to make it clearer what fields belong together may be a good option.


## References
* E. Gamma, R. Helm, R. Johnson, J. Vlissides, *Design patterns: elements of reusable object-oriented software*. Addison-Wesley Longman Publishing Co., Inc., Boston, MA, 1995.
* W. C. Wake, *Refactoring Workbook*. Addison-Wesley Professional, 2004.
