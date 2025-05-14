# Avoid floats in for loops
This rule finds `float` variables being used as loop counter. `float` values are prone to rounding and truncation. In particular, very large and very small float values are prone to rounding errors and could lead to unexpected loop behavior.


## Recommendation
Use an integral variable instead of a float variable for the loop counter.


## Example

```cpp
void f() {
	float i = 0.0f;
	//wrong: float used as loop counter
	for (i = 0; i < 1000000.0f; i++) { //may execute 1000000 +x/-x times
		//...
	}
	for (i = 0; i < 100000000.0f; i++) { //may never terminate, as rounding errors 
	                                     //cancel out the addition of 1.0 once 
	                                     //i becomes large enough
		//...
	}
}

```

## References
* AV Rule 197, *Joint Strike Fighter Air Vehicle C++ Coding Standards*. Lockheed Martin Corporation, 2005.
* MISRA C++ Rule 6-5-1, *Guidelines for the use of the C++ language in critical systems*. The Motor Industry Software Reliability Associate, 2008.
* [FLP30-C. Do not use floating-point variables as loop counters](https://www.securecoding.cert.org/confluence/display/c/FLP30-C.+Do+not+use+floating-point+variables+as+loop+counters)
