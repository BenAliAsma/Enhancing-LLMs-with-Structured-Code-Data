# TODO comment
The indicated comment is a TODO comment. TODO comments are often used to indicate code that is incomplete. This may be necessary during the implementation of new functionality but TODO comments should not be present in stable code. Any TODO comments should be reviewed and the code completed as soon as possible to avoid the accumulation of partially implemented features.


## Recommendation
Implement the functionality indicated by the comment. If the comment no longer applies, delete it to avoid confusion.


## Example

```cpp
int isOdd(int n) {
	//TODO: Works only for positive n. Need to check if negative n is valid input
	return (n % 2) == 1;
}
```

## References
* [Wikipedia: Comment tags](http://en.wikipedia.org/wiki/Comment_%28computer_programming%29#Tags)
* [TODO or not TODO](http://www.approxion.com/?p=39)
* [The case against TODO](http://wordaligned.org/articles/todo)
* Common Weakness Enumeration: [CWE-546](https://cwe.mitre.org/data/definitions/546.html).
