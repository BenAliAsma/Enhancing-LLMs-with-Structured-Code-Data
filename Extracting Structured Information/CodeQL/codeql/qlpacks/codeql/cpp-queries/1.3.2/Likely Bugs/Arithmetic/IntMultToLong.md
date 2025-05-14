# Multiplication result converted to larger type
This rule finds code that converts the result of an integer multiplication to a larger type. Since the conversion applies *after* the multiplication, arithmetic overflow may still occur.

The rule flags every multiplication of two non-constant integer expressions that is (explicitly or implicitly) converted to a larger integer type. The conversion is an indication that the expression would produce a result that would be too large to fit in the smaller integer type.


## Recommendation
Use a cast to ensure that the multiplication is done using the larger integer type to avoid overflow.


## Example

```cpp
int i = 2000000000;
long j = i * i; //Wrong: due to overflow on the multiplication between ints, 
                //will result to j being -1651507200, not 4000000000000000000

long k = (long) i * i; //Correct: the multiplication is done on longs instead of ints, 
                       //and will not overflow

long l = static_cast<long>(i) * i; //Correct: modern C++

```

## References
* MSDN Library: [Multiplicative Operators and the Modulus Operator](https://docs.microsoft.com/en-us/cpp/cpp/multiplicative-operators-and-the-modulus-operator).
* Cplusplus.com: [Integer overflow](http://www.cplusplus.com/articles/DE18T05o/).
* Common Weakness Enumeration: [CWE-190](https://cwe.mitre.org/data/definitions/190.html).
* Common Weakness Enumeration: [CWE-192](https://cwe.mitre.org/data/definitions/192.html).
* Common Weakness Enumeration: [CWE-197](https://cwe.mitre.org/data/definitions/197.html).
* Common Weakness Enumeration: [CWE-681](https://cwe.mitre.org/data/definitions/681.html).
