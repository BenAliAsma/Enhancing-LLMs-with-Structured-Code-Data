# Assignment where comparison was intended
This rule finds uses of the assignment operator `=` in places where the equality operator `==` would make more sense. This is a very common mistake in C and C++, because of the similarity of the `=` and the `==` operator, and the fact that the `if` statement accepts a condition with an integral type, instead of limiting it to just the `bool` type.

The rule flags every occurrence of an assignment in a position where its result is interpreted as a truth value. An assignment is only flagged if its right hand side is a compile-time constant.


## Recommendation
Check to ensure that the flagged expressions are not typos. If an assignment is really intended to be treated as a truth value, it may be better to surround it with parentheses.


## Example

```cpp
if(p = NULL) { //most likely == was intended. Otherwise it evaluates to the value
               //of the rhs of the assignment (which is NULL)
 ...
}

```

## References
* Tutorialspoint - The C++ Programming Language: [Operators in C++](http://www.tutorialspoint.com/cplusplus/cpp_operators.htm)
* Wikipedia: [Operators in C and C++](http://en.wikipedia.org/wiki/Operators_in_C_and_C%2B%2B#Comparison_operators.2Frelational_operators)
* Common Weakness Enumeration: [CWE-481](https://cwe.mitre.org/data/definitions/481.html).
