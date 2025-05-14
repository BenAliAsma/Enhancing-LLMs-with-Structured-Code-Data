# Unsigned difference expression compared to zero
This rule finds relational comparisons between the result of an unsigned subtraction and the value `0`. Such comparisons are likely to be wrong as the value of an unsigned subtraction can never be negative. So the relational comparison ends up checking whether the result of the subtraction is equal to `0`. This is probably not what the programmer intended.


## Recommendation
If a relational comparison is intended, consider casting the result of the subtraction to a signed type. If the intention was to test for equality, consider replacing the relational comparison with an equality test.


## Example

```c
uint32_t limit = get_limit();
uint32_t total = 0;

while (limit - total > 0) { // BAD: if `total` is greater than `limit` this will underflow and continue executing the loop.
  total += get_data();
}

while (total < limit) { // GOOD: never underflows here because there is no arithmetic.
  total += get_data();
}

while ((int64_t)limit - total > 0) { // GOOD: never underflows here because the result always fits in an `int64_t`.
  total += get_data();
}

```

## References
* SEI CERT C Coding Standard: [INT02-C. Understand integer conversion rules](https://wiki.sei.cmu.edu/confluence/display/c/INT02-C.+Understand+integer+conversion+rules).
* Common Weakness Enumeration: [CWE-191](https://cwe.mitre.org/data/definitions/191.html).
