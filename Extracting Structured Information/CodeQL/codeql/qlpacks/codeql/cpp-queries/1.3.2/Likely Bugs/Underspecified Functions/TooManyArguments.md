# Call to function with extraneous arguments
A function is called with more arguments than there are parameters of the function.

This may indicate that an incorrect function is being called, or that the signature (parameter list) of the called function is not known to the author.

In C, function calls generally need to provide the same number of arguments as there are arguments to the function. (Variadic functions can accept additional arguments.) Providing more arguments than there are parameters incurs an unneeded computational overhead, both in terms of time and of additional stack space.


## Recommendation
Call the function with the correct number of arguments.


## Example

```c
void one_argument();

void calls() {
	
	one_argument(1); // GOOD: `one_argument` will accept and use the argument
	
	one_argument(1, 2); // BAD: `one_argument` will use the first argument but ignore the second
}

void one_argument(int x);

```

## References
* SEI CERT C Coding Standard: [ DCL20-C. Explicitly specify void when a function accepts no arguments ](https://wiki.sei.cmu.edu/confluence/display/c/DCL20-C.+Explicitly+specify+void+when+a+function+accepts+no+arguments)
