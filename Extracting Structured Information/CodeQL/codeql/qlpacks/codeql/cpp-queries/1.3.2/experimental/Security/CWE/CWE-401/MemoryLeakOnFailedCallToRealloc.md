# Memory leak on failed call to realloc
Memory leak on failed call to realloc. The expression `mem = realloc (mem, size)` is potentially dangerous, if the call fails, we will lose the pointer to the memory block. An unsuccessful call is possible not only when trying to allocate a large amount of memory, but also when the process memory is strongly segmented.

False positives include code in which immediately after calling the realloc function, the pointer is manipulated without first checking for validity. In this case, an exception will occur in the program and it will terminate. But from the point of view of safe coding, these places require the attention of developers. At this stage, false positives are also possible in situations where the exception handling is quite complicated and occurs outside the base block in which memory is redistributed.


## Recommendation
We recommend storing the result in a temporary variable and eliminating memory leak.


## Example
The following example demonstrates an erroneous and corrected use of the `realloc` function.


```c
// BAD: on unsuccessful call to realloc, we will lose a pointer to a valid memory block
if (currentSize < newSize)
{
	buffer = (unsigned char *)realloc(buffer, newSize);
}



// GOOD: this way we will exclude possible memory leak 
unsigned char * tmp;
if (currentSize < newSize)
{
	tmp = (unsigned char *)realloc(buffer, newSize);
}
if (tmp == NULL)
{
	free(buffer);
} 
else
	buffer = tmp;

```

## References
* CERT C++ Coding Standard: [MEM51-CPP. Properly deallocate dynamically allocated resources](https://wiki.sei.cmu.edu/confluence/display/cplusplus/MEM51-CPP.+Properly+deallocate+dynamically+allocated+resources).
* CERT C Coding Standard: [WIN30-C. Properly pair allocation and deallocation functions](https://wiki.sei.cmu.edu/confluence/display/c/WIN30-C.+Properly+pair+allocation+and+deallocation+functions).
* Common Weakness Enumeration: [CWE-401](https://cwe.mitre.org/data/definitions/401.html).
