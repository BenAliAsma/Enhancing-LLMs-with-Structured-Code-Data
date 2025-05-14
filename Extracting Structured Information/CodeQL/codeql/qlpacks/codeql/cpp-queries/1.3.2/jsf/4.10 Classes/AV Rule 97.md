# No raw arrays in interfaces
This rule finds class members (functions or data) that are or use arrays. This is particularly important for functions with array type parameters, as these parameters are treated as pointers to the array's first element inside the function (array decay). Assuming that it is still has the type of the array passed to the function can cause unexpected behavior (e.g. when using the `sizeof` operator).


## Recommendation
Use the `Array` class, or explicitly declare the variable/parameter as a pointer so there is no possibility for confusion.


## Example

```cpp
void f(char buf[]) { //wrong: uses an array as a parameter type
	int length = sizeof(buf); //will return sizeof(char*), not the size of the array passed
	...
}

```

## References
* AV Rule 97, *Joint Strike Fighter Air Vehicle C++ Coding Standards*. Lockheed Martin Corporation, 2005.
* [comp.lang.c FAQ list · Question 6.3 ](http://c-faq.com/aryptr/aryptrequiv.html)
