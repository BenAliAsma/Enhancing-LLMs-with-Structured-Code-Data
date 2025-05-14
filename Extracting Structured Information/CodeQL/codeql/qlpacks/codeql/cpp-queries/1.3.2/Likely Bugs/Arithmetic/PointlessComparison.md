# Comparison result is always the same
Comparison operations like `x >= y` or `x != y` will always return the same result if the ranges of `x` and `y` do not overlap. In some cases this can cause an infinite loop. In the example below the loop condition on line 9 is always true because the range of `i` is \[0..5\], so the loop will never terminate.

The bounds which were deduced for the left and right operands of the comparison are included in the message as they often make it easier to understand why a result was reported. For example the message for the comparison `x >= y` might read: "Comparison is always false because x &gt;= 5 and 3 &gt;= y."


## Recommendation
Check the expression to see whether a different semantics was intended.


## Example

```cpp
int f() {
  int i;
  int total = 0;

  for (i = 0; i < 10; i = i+1) {  // GOOD: comparison could be either true or false.
    total += i;
  }

  for (i = 0; i < 10; i = i+1) {  // BAD: comparison is always true, because i <= 5. 
    i = i % 5;
    total += i;
  }

  return total;
}

```

## References
