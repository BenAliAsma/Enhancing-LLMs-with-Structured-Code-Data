# Sign check of bitwise operation
This rule finds code that checks the sign of the result of a bitwise operation. Such a check may yield unexpected results. As an example, consider the following code that checks if the `n`th bit of a variable `x` is set:

```
  x & (1 << n) > 0 
```
If `x` is a 32-bit signed integer, the value of `x & (1 << 31)` is interpreted as a signed number. If `x` is negative (that is, its sign bit is set), and `n` is 31, then `x & (1 << 31)` evaluates to `0x80000000` (all bits zero except the sign bit). The sign check on this value fails, implying that the 31st bit of `x` is unset. This is clearly incorrect.


## Recommendation
The above sign check should be rewritten as

```
  x & (1 << n) != 0
```

## References
* Code Project: [An introduction to bitwise operators](http://www.codeproject.com/Articles/2247/An-introduction-to-bitwise-operators)
* MSDN Library: [Signed Bitwise Operations](https://docs.microsoft.com/en-us/cpp/c-language/signed-bitwise-operations)
