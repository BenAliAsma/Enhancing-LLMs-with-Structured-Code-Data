# Negation of unsigned value
This query is part of a suite that tests code against the *Joint Strike Fighter Air Vehicle C++ Coding Standard* (JSF). Alerts reported by this query highlight code that may break the JSF rule listed in the References section.

The JSF rule this query tests is likely to be too strict for projects that do not follow the JSF standard.

This query finds unsigned values that are being negated. Behavior is undefined in such cases. Negating integer values produces the two's complement of that number, which cannot represent negative values of large unsigned values (values where the sign bit is used) and are most likely to be interpreted as a smaller positive integer instead.


## Recommendation
Do not negate unsigned values.


## Example

```cpp
//for this example, sizeof(short) == 2 bytes
short f(unsigned short c) {
	return -c; //unsigned value being negated
}

cout << f(100); //as expected, returns -100
cout << f(40000U); //returns 25536
// (negation finds the two's complement of the bit representation of 40000)

```

## References
* AV Rule 165, *Joint Strike Fighter Air Vehicle C++ Coding Standards*. Lockheed Martin Corporation, 2005.
