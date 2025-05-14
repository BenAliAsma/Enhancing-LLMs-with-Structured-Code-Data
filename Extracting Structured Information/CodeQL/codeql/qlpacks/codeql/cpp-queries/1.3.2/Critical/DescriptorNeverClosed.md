# Open descriptor never closed
This rule finds calls to `socket` where there is no corresponding `close` call in the program analyzed. Leaving descriptors open will cause a resource leak that will persist even after the program terminates.

> WARNING: This check is an approximation, so some results may not be actual defects in the program. It is not possible in general to compute the exact value of the variable without running the program with all possible input data.

## Recommendation
Ensure that all socket descriptors allocated by the program are freed before it terminates.


## Example
In the example below, the `sockfd` socket remains open when the `main` program finishes. The code should be updated to ensure that the socket is always closed when the program terminates.


```cpp
int main(int argc, char* argv[]) {
	int sockfd = socket(AF_INET, SOCK_STREAM, 0);
	int status = 0;
	... //code that does not close sockfd
	return status; //sockfd is never closed
}

```

## References
* SEI CERT C++ Coding Standard: [ERR57-CPP. Do not leak resources when handling exceptions](https://wiki.sei.cmu.edu/confluence/display/cplusplus/ERR57-CPP.+Do+not+leak+resources+when+handling+exceptions).
* Common Weakness Enumeration: [CWE-775](https://cwe.mitre.org/data/definitions/775.html).
