# Hiding identifiers
This query is part of a suite that tests code against the *Joint Strike Fighter Air Vehicle C++ Coding Standard* (JSF). Alerts reported by this query highlight code that may break the JSF rule listed in the References section.

The JSF rule this query tests is likely to be too strict for projects that do not follow the JSF standard.

This query highlights identifiers in an inner scope that hide (have the same name as) an identifier in an outer scope. This should be avoided as it can cause confusion about the actual variable being used in an expression.


## Recommendation
Change the name of the identifier so it does not hide another on an outer scope.


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
* AV Rule 135, *Joint Strike Fighter Air Vehicle C++ Coding Standards*. Lockheed Martin Corporation, 2005.
