# Useless parameter
Parameters that are never read in the body of the method, and are not required due to overriding, are useless and can be removed. Useless parameters unnecessarily complicate the interface for that method, and cause a maintenance and development burden.

Methods with useless parameters indicate that either the method can be simplified by removing the parameter, or that the method is not using a value it should be using. Parameters of methods that override other methods will not be marked as useless, because they are required. Similarly, parameters of methods that are overridden by other methods are not marked as useless if they are used by one of the overriding methods.


## Recommendation
The method should be inspected to determine whether the parameter should be used within the body. If the method is overridden, also consider whether any override methods should be using the parameter. If the parameter is not required, it should be removed.


## Example
In the following example, we have a method for determining whether a `String` path is an absolute path:


```java
public void isAbsolutePath(String path, String name) {
	return path.startsWith("/") || path.startsWith("\\");
}
```
The method uses the parameter `path` to determine the return value. However, the parameter `name` is not used within the body of the method. The parameter will be marked as useless, and can be removed from the program.


## References
* Wikipedia: [Redundant code](https://en.wikipedia.org/wiki/Redundant_code).
* CERT Java Coding Standard: [MSC56-J. Detect and remove superfluous code and values](https://www.securecoding.cert.org/confluence/display/java/MSC56-J.+Detect+and+remove+superfluous+code+and+values).
* Common Weakness Enumeration: [CWE-561](https://cwe.mitre.org/data/definitions/561.html).
