# Feature envy
This rule finds functions that use more functions and variables from another file than functions and variables from its own file. This function may be better placed in the other file, to avoid exposing internals of the file it depends on.


## Recommendation
See if the function can be moved to the file which contains most of its dependencies.


## References
* W. C. Wake, *Refactoring Workbook*, pp. 95 &ndash; 96. Addison-Wesley Professional, 2004.
* E. Gamma, R. Helm, R. Johnson, J. Vlissides, *Design patterns: elements of reusable object-oriented software*. Addison-Wesley Longman Publishing Co., Inc. Boston, MA, 1995.
* MSDN Magazine: [Patterns in Practice: Cohesion And Coupling](http://msdn.microsoft.com/en-us/magazine/cc947917.aspx)
