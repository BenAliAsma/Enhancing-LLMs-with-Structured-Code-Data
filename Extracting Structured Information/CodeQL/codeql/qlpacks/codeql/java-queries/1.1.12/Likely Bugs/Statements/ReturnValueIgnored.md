# Method result ignored
If the result of a method call is used in most cases, any calls to that method where the result is ignored are inconsistent, and may be erroneous uses of the API. Often, the result is some kind of status indicator, and is therefore important to check.


## Recommendation
Ensure that the results of *all* calls to a particular method are used. The return value of a method that returns a status value should normally be checked before any modified data or allocated resource is used.


## Example
Line 1 of the following example shows the value returned by `get` being ignored. Line 3 shows it being assigned to `fs`.


```java
FileSystem.get(conf);  // Return value is not used

FileSystem fs = FileSystem.get(conf);  // Return value is assigned to 'fs'
```

## References
* CERT Secure Coding Standards: [ EXP00-J. Do not ignore values returned by methods](https://www.securecoding.cert.org/confluence/display/java/EXP00-J.+Do+not+ignore+values+returned+by+methods).
* Common Weakness Enumeration: [CWE-252](https://cwe.mitre.org/data/definitions/252.html).
