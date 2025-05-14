# Non-private field
A non-final or non-static field that is not declared `private`, but is not accessed outside of its declaring type, may decrease code maintainability. This is because a field that is accessible from outside the class that it is declared in tends to restrict the class to a particular implementation.


## Recommendation
In the spirit of encapsulation, it is generally advisable to choose the most restrictive access modifier (`private`) for a field, unless there is a good reason to increase its visibility.


## References
* J. Bloch, *Effective Java (second edition)*, Item 13. Addison-Wesley, 2008.
* The Java Tutorials: [Controlling Access to Members of a Class](https://docs.oracle.com/javase/tutorial/java/javaOO/accesscontrol.html).
