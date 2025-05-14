# FIXME comment
The indicated comment is a FIXME comment. FIXME comments are often used to indicate code that does not work correctly or that may not work in all supported environments. This may be necessary during the implementation of new functionality but FIXME comments should not be present in stable code. Any FIXME comments should be reviewed and the code improved as soon as possible to avoid the accumulation of partially implemented features.


## Recommendation
Fix the functionality indicated by the comment. If the comment no longer applies, delete it to avoid confusion.


## Example

```cpp
int isEven(int n) {
	//FIXME: Is only correct for small values of n
	return n == 0 || n == 2;
}
```

## References
* [Wikipedia: Comment tags](http://en.wikipedia.org/wiki/Comment_%28computer_programming%29#Tags)
* [The case against TODO (and FIXME)](http://wordaligned.org/articles/todo)
* Common Weakness Enumeration: [CWE-546](https://cwe.mitre.org/data/definitions/546.html).
