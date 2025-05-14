# Missed opportunity to call wrapper function
The presence of a shell function with additional check indicates the possible risks of the call. Use this check everywhere.


## Example
The following example demonstrates fallacious and fixed methods of using wrapper functions.


```cpp
...
int myFclose(FILE * fmy)
{
  if(!fclose(fmy)) {
    fmy = NULL;
    return 0;
  }
  return -1;
}
...
  fe = fopen("myFile.txt", "wt"); 
  ...
  fclose(fe); // BAD
...
  fe = fopen("myFile.txt", "wt"); 
  ...
  myFclose(fe); // GOOD
...

```

## References
* CERT C Coding Standard: [JNI00-J. Define wrappers around native methods](https://wiki.sei.cmu.edu/confluence/display/java/JNI00-J.+Define+wrappers+around+native+methods).
* Common Weakness Enumeration: [CWE-1041](https://cwe.mitre.org/data/definitions/1041.html).
