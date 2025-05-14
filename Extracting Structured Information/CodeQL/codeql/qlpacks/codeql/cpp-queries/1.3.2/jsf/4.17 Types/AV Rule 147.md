# Do not expose float representation
This query is part of a suite that tests code against the *Joint Strike Fighter Air Vehicle C++ Coding Standard* (JSF). Alerts reported by this query highlight code that may break the JSF rule listed in the References section.

The JSF rule this query tests is likely to be too strict for projects that do not follow the JSF standard.

This query highlights portions of code that can expose the floating point implementation of the underlying machine. Manually manipulating the bits in the float is prone to mistakes and is unportable. Floating point implementations can vary across architectures, and bit-field packing can differ across compilers, making manual bit-manipulation of floats inadvisable.

The bits of a floating point could be exposed by:

* casting a float pointer to a pointer of another type
* casting a float array to a non-float pointer type
* using a float in a union with another type

## Recommendation
Do not expose the bit contents of a float.


## Example

```cpp
// Wrong: floating point bit implementation exposed by union with bit field.
// Endianness and different floating-point implementations across architectures
// as well as different packing methods across compilers could make this behave
// incorrectly.
typedef union {
    float float_num;
    struct {
        unsigned sign : 1;
        unsigned exp : 8;
        unsigned fraction : 23;
    } bits;
} floatbits;

```

## References
* AV Rule 147, *Joint Strike Fighter Air Vehicle C++ Coding Standards*. Lockheed Martin Corporation, 2005.
