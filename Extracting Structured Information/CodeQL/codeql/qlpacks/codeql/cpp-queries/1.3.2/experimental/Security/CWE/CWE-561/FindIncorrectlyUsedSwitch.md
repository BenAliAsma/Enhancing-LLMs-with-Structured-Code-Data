# Incorrect switch statement
A mismatch between conditionals and `switch` cases can lead to control-flow violations (CWE-691) where the developer either does not handle all combinations of conditions or unintentionally created dead code (CWE-561).


## Example
The following example demonstrates fallacious and fixed ways of using a `switch` statement.


```c
...
  int i1;
  char c1;
...
  if((c1<50)&&(c>10))
  switch(c1){
    case 300: // BAD: the code will not be executed
...  
  if((i1<5)&&(i1>0))
  switch(i1){ // BAD
    case 21: // BAD: the code will not be executed
...
  switch(c1){
...
  dafault: // BAD: maybe it will be right `default`
...
  }

...
  switch(c1){ 
      i1=c1*2; // BAD: the code will not be executed
    case 12:
...
  switch(c1){ // GOOD
    case 12:
      break;
    case 10:
      break;
    case 9:
      break;
    default:
      break;
  }
...

```

## References
* CERT C Coding Standard: [MSC12-C. Detect and remove code that has no effect or is never executed](https://wiki.sei.cmu.edu/confluence/display/c/MSC12-C.+Detect+and+remove+code+that+has+no+effect+or+is+never+executed).
* Common Weakness Enumeration: [CWE-561](https://cwe.mitre.org/data/definitions/561.html).
* Common Weakness Enumeration: [CWE-691](https://cwe.mitre.org/data/definitions/691.html).
* Common Weakness Enumeration: [CWE-478](https://cwe.mitre.org/data/definitions/478.html).
