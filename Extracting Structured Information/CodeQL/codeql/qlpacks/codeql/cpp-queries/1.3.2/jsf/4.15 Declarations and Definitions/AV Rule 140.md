# Register variables
This query is part of a suite that tests code against the *Joint Strike Fighter Air Vehicle C++ Coding Standard* (JSF). Alerts reported by this query highlight code that may break the JSF rule listed in the References section.

The JSF rule this query tests is likely to be too strict for projects that do not follow the JSF standard.

This query highlights variables with the `register` storage class specifier. Modern compilers are now capable of optimal register placement, and overriding it could lead to worse performance.


## Recommendation
Remove the `register` storage class specifier.


## Example

```cpp
int f() {
	//wrong: register storage specifier used
	register int i; //these register definitions can crowd out the
	                //variables x1 to x5 that are used in the complex
	                //operation in the inner loop, leading to slower
	                //performance
	register int j;

	for (i = 0; i < HUGE_NUM; i++) {
		for (j = 0; j < HUGE_NUM; j++) {
			int x1, x2, x3, x4, x5;
			//complex CPU-intensive operation that accesses
			//x1 to x5 a large number of times
		}
	}
}

```

## References
* AV Rule 140, *Joint Strike Fighter Air Vehicle C++ Coding Standards*. Lockheed Martin Corporation, 2005.
* M. Banahan, D. Brady, M. Doram. *The C Book*. Section 8.2.1.1. `http://publications.gbdirect.co.uk/c_book/`
