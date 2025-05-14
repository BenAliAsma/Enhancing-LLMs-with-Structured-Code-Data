# Variable used in its own initializer
A variable is in scope in its own initializer, but it is undefined behavior to load from it before it is first assigned to.


## Recommendation
Do not use a variable in its own initializer unless it is part of an address calculation or a `sizeof` expression.


## Example

```cpp
int f() {
	int x = x; // BAD: undefined behavior occurs here
	x = 0;
	return x;
}

int g() {
	int x = 0; // GOOD
	return x;
}

```

## References
* [ SEI CERT Secure Coding Standard: EXP53-CPP. Do not read uninitialized memory ](https://wiki.sei.cmu.edu/confluence/display/cplusplus/EXP53-CPP.+Do+not+read+uninitialized+memory)
