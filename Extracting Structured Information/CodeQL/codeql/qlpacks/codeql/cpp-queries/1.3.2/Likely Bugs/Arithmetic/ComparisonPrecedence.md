# Unclear comparison precedence
This rule finds comparison expressions that use 2 or more comparison operators and are not completely parenthesized. It is best to fully parenthesize complex comparison expressions to explicitly define the order of the comparison operators.


## Recommendation
Fully parenthesize complex comparison expressions to avoid confusion.


## Example

```cpp
void h() {
	int a, b, c;

	a < b != c; //parenthesize to explicitly define order of operators
	(a < b) < c; //correct: parenthesized to specify order
}

```

## References
* MSDN Library: [C++ built-in operators, precedence, and associativity](https://docs.microsoft.com/en-us/cpp/cpp/cpp-built-in-operators-precedence-and-associativity)
* [Operators](http://www.cplusplus.com/doc/tutorial/operators/)
* [EXP00-C. Use parentheses for precedence of operation](https://wiki.sei.cmu.edu/confluence/display/c/EXP00-C.+Use+parentheses+for+precedence+of+operation)
