# Open descriptor may not be closed
This query looks at functions that return file or socket descriptors, but may return an error value before actually closing the resource. This can occur when an operation performed on the open descriptor fails, and the function returns with an error before it closes the open resource. An improperly handled error could cause the function to leak resource descriptors. Failing to close resources in the function that opened them also makes it more difficult to detect leaks.

> WARNING: This check is an approximation, so some results may not be actual defects in the program. It is not possible in general to compute the actual branch taken in conditional statements such as "if" without running the program with all possible input data. This means that it is not possible to determine if a particular statement is going to be executed.

## Recommendation
When an error occurs, ensure that the function frees all the resources it holds open.


## Example
In the example below, the `sockfd` socket may remain open if an error is triggered. The code should be updated to ensure that the socket is always closed when the function ends.


```cpp
int f() {
	try {
		int sockfd = socket(AF_INET, SOCK_STREAM, 0);
		do_stuff(sockfd);
		return sockfd; //if there are no exceptions, the socket is returned
	} catch (int do_stuff_exception) {
		return -1; //return error value, but sockfd may still be open
	}
}

```

## References
* SEI CERT C++ Coding Standard: [ERR57-CPP. Do not leak resources when handling exceptions](https://wiki.sei.cmu.edu/confluence/display/cplusplus/ERR57-CPP.+Do+not+leak+resources+when+handling+exceptions).
* Common Weakness Enumeration: [CWE-775](https://cwe.mitre.org/data/definitions/775.html).
