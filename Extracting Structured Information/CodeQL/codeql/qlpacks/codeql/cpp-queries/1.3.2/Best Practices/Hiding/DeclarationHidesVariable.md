# Declaration hides variable
This rule finds declarations of local variables that hide a local variable from a surrounding scope. Such declarations create variables with the same name but different scopes. This makes it difficult to know which variable is actually used in an expression.


## Recommendation
Consider changing the name of either variable to keep them distinct.


## Example

```cpp
void f() {
	int i = 10;

	for (int i = 0; i < 10; i++) { //the loop counter hides the variable
		 ...
	}

	{
		int i = 12; //this variable hides the variable in the outer block
		...
	}
}

```

## References
* B. Stroustrup. *The C++ Programming Language Special Edition* p 82. Addison Wesley. 2000.
