# Local variable hides global variable
This rule finds declarations of local variables or parameters that hide a global variable. Such declarations create variables with the same name but different scopes. This makes it difficult to know which variable is actually used in an expression.


## Recommendation
Consider changing the name of either variable to keep them distinct.


## Example

```cpp
int i = 10;

void f() {
	for (int i = 0; i < 10; i++) { //the loop counter hides the global variable i
		 ...
	}

	{
		int i = 12; //this variable hides the global variable i
		...
	}
}

```

## References
* B. Stroustrup. *The C++ Programming Language Special Edition* p 82. Addison Wesley. 2000.
