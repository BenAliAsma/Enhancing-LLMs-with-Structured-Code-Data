# Branching condition always evaluates to same value
This query finds branching statements with conditions that always evaluate to the same value. It is likely that these conditions indicate an error in the branching condition. Alternatively, the conditions may have been left behind after debugging.

> WARNING: This check is an approximation, so some results may not be actual defects in the program. It is not possible in general to compute the exact value of the variable without running the program with all possible input data.

## Recommendation
Check the branch condition for logic errors. Check whether it is still required.


## Example
This example shows two branch conditions that always evaluate to the same value. The two conditions and their associated branches should be deleted. This will simplify the code and make it easier to maintain.


```cpp
while(result) {
	if ( ... )
		...
	else if (result //wrong: this test is redundant
				&& result->flags != 0)
		...
	result = next(queue);
}


fp = fopen(log, "r");
if (fp) {
	/*
	 * large block of code
	 */
	if (!fp) { //wrong: always false
		...  /* dead code */
	}
}

```

## References
* SEI CERT C++ Coding Standard [MSC12-C. Detect and remove code that has no effect or is never executed](https://wiki.sei.cmu.edu/confluence/display/c/MSC12-C.+Detect+and+remove+code+that+has+no+effect+or+is+never+executed).
* Common Weakness Enumeration: [CWE-561](https://cwe.mitre.org/data/definitions/561.html).
