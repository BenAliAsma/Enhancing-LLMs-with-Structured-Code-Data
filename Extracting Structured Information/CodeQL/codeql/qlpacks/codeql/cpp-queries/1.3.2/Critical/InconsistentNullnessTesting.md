# Inconsistent null check of pointer
This query finds pointer dereferences that do not first check the pointer for nullness, even though the same pointer is checked for nullness in other parts of the code. It is likely that the nullness check was accidentally omitted, and that a null pointer dereference can occur. Dereferencing a null pointer and attempting to modify its contents can lead to anything from a segmentation fault to corrupting important system data (including the interrupt table in some architectures).

> WARNING: This check is an approximation, so some results may not be actual defects in the program. It is not possible in general to compute the values of pointers without running the program with all input data.

## Recommendation
Use a nullness check consistently in all cases where a pointer is dereferenced.


## Example
This code shows two examples where a pointer is dereferenced. The first example checks that the pointer is not null before dereferencing it. The second example fails to perform a nullness check, leading to a potential vulnerability in the code.


```cpp
void* f() {
	block = (MyBlock *)malloc(sizeof(MyBlock));
	if (block) { //correct: block is checked for nullness here
		block->id = NORMAL_BLOCK_ID;
	}
	//...
	/* make sure data-portion is null-terminated */
	block->data[BLOCK_SIZE - 1] = '\0'; //wrong: block not checked for nullness here
	return block;
}

```

## References
* SEI CERT C++ Coding Standard: [MEM10-C. Define and use a pointer validation function](https://wiki.sei.cmu.edu/confluence/display/c/MEM10-C.+Define+and+use+a+pointer+validation+function).
* Common Weakness Enumeration: [CWE-476](https://cwe.mitre.org/data/definitions/476.html).
