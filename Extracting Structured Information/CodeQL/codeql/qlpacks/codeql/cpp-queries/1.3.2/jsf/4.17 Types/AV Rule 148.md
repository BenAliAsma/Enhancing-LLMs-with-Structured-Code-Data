# Use of integer where enum is preferred
This rule finds `switch` statements that use an integer instead of an enumeration. Enumerations are preferred when dealing with a limited number of choices as they makes it easier to see if a case has been left out.


## Recommendation
Use an enumeration instead of an integer to represent a limited set of choices.


## Example

```cpp
typedef enum {
	CASE_VAL1,
	CASE_VAL2
} caseVals;

void f() {
	int caseVal;
	//Wrong: switch statement uses an integer
	switch(caseVal) {
	case 1:
		//...
	case 0xFF:
		//...
	default:
		//...
	}

	//Correct: switch statement uses enum. It is easier to see if a case 
	//has been left out, and that all cases are valid values
	caseVals caseVal2;
	switch (caseVal2) {
	case CASE_VAL1:
		//...
	case CASE_VAL2:
		//...
	default:
	}
}

```

## References
* AV Rule 148, *Joint Strike Fighter Air Vehicle C++ Coding Standards*. Lockheed Martin Corporation, 2005.
* [C++ Switch statement](http://www.tutorialspoint.com/cplusplus/cpp_switch_statement.htm)
