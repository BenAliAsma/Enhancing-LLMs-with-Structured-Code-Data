# Overloaded assignment does not return 'this'
This rule finds assignment operator definitions that do not return a reference to `this`. The reference to `this` is required for an assignment expression to have a meaningful value (i.e. so that expressions like `a = b = 3` would work correctly). Not returning a reference to `this` would give the assignment expression an unexpected value, most likely causing such chained assignments to fail. Both the standard library types and the built-in types behave in this manner, and the default behavior of the assignment operator is an almost universal assumption of all developers that it is unwise to change it considerably. If the desired behavior is significantly different from that of the default assignment operator, it may be better to define a separate function instead.


## Recommendation
Make sure that the assignment operator overload returns a reference to `this`. Define a separate function if the desired assignment operator differs significantly from the default behavior.


## Example

```cpp
class B {
	B& operator=(const B& other) {
		... //incorrect, does not return a reference to this
	}
};

class C {
	C& operator=(const C& other) {
		...
		return *this; //correct, returns reference to this
	}
};

```

## References
* AV Rule 82, *Joint Strike Fighter Air Vehicle C++ Coding Standards*. Lockheed Martin Corporation, 2005.
* S. Meyers. *Effective C++ 3d ed.* pp 52-53. Addison-Wesley Professional, 2005.
* [C++ Operator Overloading Guidelines](http://courses.cms.caltech.edu/cs11/material/cpp/donnie/cpp-ops.html)
