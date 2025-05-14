# Confusing overloading of methods
Overloaded method declarations that have the same number of parameters may be confusing if none of the corresponding pairs of parameter types is substantially different. A pair of parameter types A and B is substantially different if A cannot be cast to B and B cannot be cast to A. If the parameter types are not substantially different then the programmer may assume that the method with parameter type A is called when in fact the method with parameter type B is called.


## Recommendation
It is generally best to avoid declaring overloaded methods with the same number of parameters, unless at least one of the corresponding parameter pairs is substantially different.


## Example
Declaring overloaded methods `process(Object obj)` and `process(String s)` is confusing because the parameter types are not substantially different. It is clearer to declare methods with different names: `processObject(Object obj)` and `processString(String s)`.

In contrast, declaring overloaded methods `process(Object obj, String s)` and `process(String s, int i)` is not as confusing because the second parameters of each method are substantially different.


## References
* J. Bloch, *Effective Java (second edition)*, Item 41. Addison-Wesley, 2008.
* Java Language Specification: [15.12 Method Invocation Expressions](https://docs.oracle.com/javase/specs/jls/se11/html/jls-15.html#jls-15.12).
