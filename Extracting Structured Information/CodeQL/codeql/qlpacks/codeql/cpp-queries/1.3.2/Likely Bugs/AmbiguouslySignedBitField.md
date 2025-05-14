# Ambiguously signed bit-field member
The signedness of a plain char, short, int, or long bit field is implementation-specific in C and in older versions of C++, and declaring their signedness explicitly removes the ambiguity and ensures portability.


## Recommendation
Declare all members of the bit field with explicit signedness.


## Example

```cpp
struct {
	int s : 4; //wrong: behavior of bit-field members with implicit signage vary across compilers
	unsigned int : 24; //correct: explicitly unsigned
	signed int : 4; //correct: explicitly signed
} bits;

```

## References
* AV Rule 154, *Joint Strike Fighter Air Vehicle C++ Coding Standards*. Lockheed Martin Corporation, 2005.
* [C++ Bit Fields](http://en.cppreference.com/w/cpp/language/bit_field)
* Common Weakness Enumeration: [CWE-190](https://cwe.mitre.org/data/definitions/190.html).
