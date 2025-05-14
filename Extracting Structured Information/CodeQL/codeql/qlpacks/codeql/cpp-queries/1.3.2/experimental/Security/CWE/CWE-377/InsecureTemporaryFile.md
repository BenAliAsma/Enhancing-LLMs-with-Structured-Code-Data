# Insecure generation of filenames.
Working with a file, without checking its existence and its rights, as well as working with names that can be predicted, may not be safe. Requires the attention of developers.


## Example
The following example demonstrates erroneous and corrected work with file.


```cpp
...
  fp = fopen("/tmp/name.tmp","w"); // BAD
...
  char filename = tmpnam(NULL);
  fp = fopen(filename,"w"); // BAD
...

  strcat (filename, "/tmp/name.XXXXXX");
  fd = mkstemp(filename);
  if ( fd < 0 ) {
    return error;
  }
  fp = fdopen(fd,"w") // GOOD
...

```

## References
* CERT C Coding Standard: [CON33-C. Avoid race conditions when using library functions](https://wiki.sei.cmu.edu/confluence/display/c/CON33-C.+Avoid+race+conditions+when+using+library+functions).
* Common Weakness Enumeration: [CWE-377](https://cwe.mitre.org/data/definitions/377.html).
