# Inappropriate Intimacy
This rule looks for two files that share too much information about each other (accessing many operations or variables in both directions). It would be better to invert some of the dependencies to reduce the coupling between the two files.

Having two files have too many dependencies on each other makes it difficult to modify one file without requiring modifications to the other. This could lead to defects when the programmer forgets to put in the necessary changes to one file when he makes a change to the other.


## Recommendation
Move some of the methods and variables from one file to another, so that most of the dependencies go only in one direction. If possible, try to make all the dependencies go in one direction.


## References
* W. C. Wake, *Refactoring Workbook*, pp. 95 &ndash; 96. Addison-Wesley Professional, 2004.
* E. Gamma, R. Helm, R. Johnson, J. Vlissides, *Design patterns: elements of reusable object-oriented software*. Addison-Wesley Longman Publishing Co., Inc. Boston, MA, 1995.
* MSDN Magazine: [Patterns in Practice: Cohesion And Coupling](http://msdn.microsoft.com/en-us/magazine/cc947917.aspx)
