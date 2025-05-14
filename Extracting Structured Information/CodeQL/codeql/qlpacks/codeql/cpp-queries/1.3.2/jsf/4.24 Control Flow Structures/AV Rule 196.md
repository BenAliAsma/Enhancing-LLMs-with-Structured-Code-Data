# No trivial switch statements
The following forms of `switch` statement are considered *trivial*:

1. No cases at all.
1. Just a default case.
1. Just one non-default case.
1. A default case and one non-default case.

## Recommendation
Either the `switch` statement should be replaced with a simpler control flow structure, or it should be extended to handle more cases. Each trivial form has a different replacement:

1. If there are no cases, the `switch` statement can be removed.
1. If there is just one default case, the `switch` keyword, the `default` keyword, and the subsequent colon can all be removed.
1. If there is just one non-default case, the `switch` statement can be turned into an `if` statement.
1. If there is one default case and one non-default case, the `switch` statement can be turned into an `if`/`else` statement.

## Example

```cpp
int f() {
	int val = 0;
	switch(val) { //wrong, use an if instead
	case 0:
		//...
	default:
		//...
	}

	switch(val) { //correct, has 2 cases and a default
	case 0:
		//...
	case 1:
		//...
	default:
		//...
	}
}

```

## References
* AV Rule 196, *Joint Strike Fighter Air Vehicle C++ Coding Standards*. Lockheed Martin Corporation, 2005.
