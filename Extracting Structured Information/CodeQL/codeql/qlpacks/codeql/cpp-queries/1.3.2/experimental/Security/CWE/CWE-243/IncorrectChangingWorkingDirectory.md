# Find work with changing working directories, with security errors.
Working with changing directories, without checking the return value or pinning the directory, may not be safe. Requires the attention of developers.


## Example
The following example demonstrates erroneous and corrected work with changing working directories.


```cpp
...
  chroot("/myFold/myTmp"); // BAD
...
  chdir("/myFold/myTmp"); // BAD
...
  int fd = open("/myFold/myTmp", O_RDONLY | O_DIRECTORY);
  fchdir(fd); // BAD
...
  if (chdir("/myFold/myTmp") == -1) {
    exit(-1);
  }
  if (chroot("/myFold/myTmp") == -1) {  // GOOD
    exit(-1);
  }
...
  if (chdir("/myFold/myTmp") == -1) { // GOOD
    exit(-1);
  }
...
  int fd = open("/myFold/myTmp", O_RDONLY | O_DIRECTORY);
  if(fchdir(fd) == -1) { // GOOD
    exit(-1);
  }
...

```

## References
* CERT C Coding Standard: [POS05-C. Limit access to files by creating a jail.](https://wiki.sei.cmu.edu/confluence/display/c/POS05-C.+Limit+access+to+files+by+creating+a+jail)
* Common Weakness Enumeration: [CWE-243](https://cwe.mitre.org/data/definitions/243.html).
* Common Weakness Enumeration: [CWE-252](https://cwe.mitre.org/data/definitions/252.html).
