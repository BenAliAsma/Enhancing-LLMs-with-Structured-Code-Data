# Unsigned comparison to zero
This rule finds expressions of the form `x >= 0` where `x` is an unsigned value. This comparison is pointless as it will always yield `1`.


## Recommendation
Check the expression to see whether a different semantics was intended.


## Example

```cpp
typedef long long LONGLONG;

int f(unsigned int u, LONGLONG l) {
	if(u > 0 || l >=0)       //correct: unsigned value is check for > 0
		return 23;
	return u >= 0;           //wrong: unsigned values are always greater than or equal to 0
}

```

## References
* [Variables. Data types.](http://www.cplusplus.com/doc/tutorial/variables/)
