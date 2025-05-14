# Errors When Double Release
Double release of the descriptor can lead to a crash of the program. Requires the attention of developers.


## Recommendation
We recommend that you exclude situations of possible double release.


## Example
The following example demonstrates an erroneous and corrected use of descriptor deallocation.


```c
...
  fs = socket(AF_UNIX, SOCK_STREAM, 0)
...
  close(fs);
  fs = -1; // GOOD
...

...
  fs = socket(AF_UNIX, SOCK_STREAM, 0)
...
  close(fs);
  if(fs) close(fs); // BAD
...

```

## References
* CERT C Coding Standard: [FIO46-C. Do not access a closed file](https://wiki.sei.cmu.edu/confluence/display/c/FIO46-C.+Do+not+access+a+closed+file).
* Common Weakness Enumeration: [CWE-675](https://cwe.mitre.org/data/definitions/675.html).
* Common Weakness Enumeration: [CWE-666](https://cwe.mitre.org/data/definitions/666.html).
