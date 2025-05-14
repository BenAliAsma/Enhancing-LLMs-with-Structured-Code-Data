# Pointer offset used before it is checked
This query finds integer values that are first used to index an array and subsequently tested for being negative. If it is relevant to perform this test at all then it should happen *before* the indexing, not after. Otherwise, if the value is negative then the program will have failed before performing the test.

> WARNING: This check is an approximation, so some results may not be actual defects in the program. It is not possible in general to compute the actual branch taken in conditional statements such as "if" without running the program with all possible input data. This means that it is not possible to determine if a particular statement is going to be executed.

## Recommendation
See if the value needs to be checked before being used as array index. Double-check if the value is derived from user input. If the value clearly cannot be negative then the negativity test is redundant and can be removed.


## Example
The example below includes two functions that use the value `recordIdx` to index an array and a test to verify that the value is positive. The test is made after `records` is indexed for `printRecord` and before `records` is indexed for `processRecord`. Unless the value of `recordIdx` cannot be negative, the test should be updated to run before *both* times the array is indexed. If the value cannot be negative, the test should be removed.


```cpp
Record records[SIZE] = ...;

int f() {
	int recordIdx = 0;
	cin >> recordIdx;
	printRecord(&(records[recordIdx])); //incorrect: recordIdx may be negative here

	if (recordIdx >= 0) {
		processRecord(&(records[recordIdx])); //correct: index checked before use
	}
}

```

## References
* cplusplus.com: [Pointers](http://www.cplusplus.com/doc/tutorial/pointers/).
* SEI CERT C Coding Standard: [ARR30-C. Do not form or use out-of-bounds pointers or array subscripts](https://wiki.sei.cmu.edu/confluence/display/c/ARR30-C.+Do+not+form+or+use+out-of-bounds+pointers+or+array+subscripts).
* Common Weakness Enumeration: [CWE-823](https://cwe.mitre.org/data/definitions/823.html).
