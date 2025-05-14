# Errors When Double Free
Freeing a previously allocated resource twice can lead to various vulnerabilities in the program.


## Recommendation
We recommend that you exclude situations of possible double release. For example, use the assignment NULL to a freed variable.


## Example
The following example demonstrates an erroneous and corrected use of freeing a pointer.


```c
...
  buf = malloc(intSize);
...
  free(buf); 
  buf = NULL;
  if(buf) free(buf); // GOOD
...

...
  buf = malloc(intSize);
...
  free(buf); 
  if(buf) free(buf); // BAD: the cleanup function does not zero out the pointer
...

```

## References
* CERT C Coding Standard: [MEM30-C. Do not access freed memory](https://wiki.sei.cmu.edu/confluence/display/c/MEM30-C.+Do+not+access+freed+memory).
* Common Weakness Enumeration: [CWE-415](https://cwe.mitre.org/data/definitions/415.html).
