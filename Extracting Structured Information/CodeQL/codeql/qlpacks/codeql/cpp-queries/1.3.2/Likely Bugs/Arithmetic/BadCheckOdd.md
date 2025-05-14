# Bad check for oddness
This rule finds code that uses `x % 2 == 1` to check whether a number `x` is odd, which does not work for negative numbers. Applying `%` to negative numbers produces negative results. For example, `(-5) % 2 ` equals `-1`, not `1`. As a result, this check incorrectly considers all negative numbers as even.


## Recommendation
Consider using `x % 2 != 0` or `(x & 1) == 1` instead.


## References
* MSDN Library: [Multiplicative Operators and the Modulus Operator](https://docs.microsoft.com/en-us/cpp/cpp/multiplicative-operators-and-the-modulus-operator).
* Wikipedia: [Modulo Operation - Common pitfalls](http://en.wikipedia.org/wiki/Modulo_operation#Common_pitfalls).
