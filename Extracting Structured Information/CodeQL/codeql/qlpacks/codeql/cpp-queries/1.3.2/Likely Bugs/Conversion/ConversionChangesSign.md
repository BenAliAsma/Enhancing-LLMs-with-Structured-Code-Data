# Conversion changes sign
This rule finds expressions which are implicitly converted from signed to unsigned or vice versa. Casting an unsigned value to a signed value and vice versa does not change the contents (in bits) of the variable, it just changes the way the sign bit is interpreted. Casting a large unsigned value (one with the most significant bit set) to a signed value will make it negative, while casting a negative signed value converts it to a large unsigned number.


## Recommendation
Make the conversion explicit by adding a cast (and comparing against zero where appropriate).


## Example

```cpp
//sz is a signed integer, but malloc expects one that is unsigned.
//Negative values will be interpreted as a large number, which may
//lead to unexpected behavior
char *buf = malloc(sz);

```
