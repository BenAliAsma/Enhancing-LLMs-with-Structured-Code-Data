# Sizeof with side effects
This rule finds `sizeof` expressions that have a parameter with side-effects. `sizeof` only uses the type of the parameter, so the parameter will not be evaluated. In the C99 standard, using `sizeof` on expressions with dynamic arrays may or may not evaluate the side-effect, so it is better to avoid it completely.


## Recommendation
Simplify the `sizeof` parameter to use only the subexpression that is of the type you need.


## Example

```cpp
int f(void){
	int i = 0;
	char arr[20];
	int size = sizeof(arr[i++]); //wrong: sizeof expression has side effect
	cout << i; //would output 0 instead of 1
}

```

## References
* AV Rule 166, *Joint Strike Fighter Air Vehicle C++ Coding Standards*. Lockheed Martin Corporation, 2005.
* Tutorialspoint - The C++ Programming Language: [C++ sizeof Operator](http://www.tutorialspoint.com/cplusplus/cpp_sizeof_operator.htm)
