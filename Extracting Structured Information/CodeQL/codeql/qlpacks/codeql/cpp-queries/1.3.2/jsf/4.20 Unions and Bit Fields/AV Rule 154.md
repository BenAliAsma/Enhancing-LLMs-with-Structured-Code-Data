# Possible signed bit-field member
This query is part of a suite that tests code against the *Joint Strike Fighter Air Vehicle C++ Coding Standard* (JSF). Alerts reported by this query highlight code that may break the JSF rule listed in the References section.

The JSF rule this query tests is likely to be too strict for projects that do not follow the JSF standard.

This query finds bit fields with members that are not explicitly declared to be unsigned. The sign of plain char, short, int, or long bit field is implementation-specific, and declaring them all to be unsigned removes the ambiguity and ensures portability.


## Recommendation
Declare all members of the bit field to be unsigned.


## Example
The code below shows two examples of bit fields. The second field is declared to be unsigned which ensures portability. The first field should also be declared to be unsigned.


```cpp
struct {
	short s : 4; //wrong: behavior of signed bit-field members varies across compilers
	unsigned int : 24; //correct: unsigned
} bits;

```

## References
* AV Rule 154, *Joint Strike Fighter Air Vehicle C++ Coding Standards*. Lockheed Martin Corporation, 2005.
* C++ reference: [Bit Fields](http://en.cppreference.com/w/cpp/language/bit_field)
* Common Weakness Enumeration: [CWE-190](https://cwe.mitre.org/data/definitions/190.html).
