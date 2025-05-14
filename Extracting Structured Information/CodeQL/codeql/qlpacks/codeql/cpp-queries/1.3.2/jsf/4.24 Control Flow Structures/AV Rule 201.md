# For loop variable changed in body
This rule finds `for` loop variables that are being modified in the loop body. `for` loops should be reserved for simple iteration, use `while` loops for more complicated cases. Most developers assume that the `for` loop iteration statement is the only thing that changes the value of the loop counter, so changing it inside the loop body can lead to confusion.


## Recommendation
Use only the increment expression in the `for` loop to modify loop variables, or change it to a `while` loop if the loop requires more complicated variable iteration.


## Example

```cpp
void f() {
	int i = 0;
	for (i = 0; i < 10; i++) {
		//...
		if (special_case) {
			i += 5; //Wrong: loop variable changed in body, which is contrary 
			        //to the usual expectation of for loop behavior. 
			        //Use a while loop instead.
			continue;
		}
	}
}

```

## References
* AV Rule 201, *Joint Strike Fighter Air Vehicle C++ Coding Standards*. Lockheed Martin Corporation, 2005.
* MISRA C++ Rule 6-5-3, *Guidelines for the use of the C++ language in critical systems*. The Motor Industry Software Reliability Associate, 2008.
* [For statements](http://www.learncpp.com/cpp-tutorial/57-for-statements/)
* Mats Henricson and Erik Nyquist, *Industrial Strength C++*, published by Prentice Hall PTR (1997). Chapter 4: Control Flow, Rule 4.1 ([PDF](https://web.archive.org/web/20190919025638/https://mongers.org/industrial-c++/)).
