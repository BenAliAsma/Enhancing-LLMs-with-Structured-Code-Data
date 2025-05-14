# Left shift by more than the type width
The maximum shift distance used for left-shift operations is determined by the promoted type of its left-hand side. When the promoted type is `int` only the lowest 5 bits of the right-hand side are used as the shift distance. When the promoted type is `long` the lowest 6 bits of the right-hand side are used.


## Recommendation
Restrict the amount that you shift any `int` to the range 0-31, or cast it to `long` before applying the left shift.


## Example
The following line tries to left-shift an `int` by 32 bits.


```java
long longVal = intVal << 32; // BAD
```
However, left-shifting an `int` by 32 bits is equivalent to left-shifting it by 0 bits, that is, no shift is applied. Instead the value should be cast to `long` before the shift is applied. Then the left-shift of 32 bits will work.


```java
long longVal = ((long)intVal) << 32; // GOOD
```

## References
* Java Language Specification: [Shift Operators](https://docs.oracle.com/javase/specs/jls/se11/html/jls-15.html#jls-15.19).
