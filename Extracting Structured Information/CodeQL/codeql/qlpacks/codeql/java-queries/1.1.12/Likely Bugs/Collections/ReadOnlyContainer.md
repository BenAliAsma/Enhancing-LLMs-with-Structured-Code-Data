# Container contents are never initialized
A method that queries the contents of a collection or map (such as `containsKey` or `isEmpty`) is invoked on an object that is known to be empty. Such method calls do not return interesting results, and may indicate missing code or a logic error.


## Recommendation
Either remove the collection/map if it is unnecessary, or ensure that it contains the elements it was meant to contain.


## Example
The following example code iterates over an array of objects to determine whether it contains duplicate elements. It maintains a collection `seen`, which is intended to contain all the elements seen so far in traversing the array. If the current element is already contained in that collection then the method returns `true`, indicating that a duplicate has been found.

Note, however, that no elements are ever actually added to `seen`, so the method always returns `false`.


```java
boolean containsDuplicates(Object[] array) {
	java.util.Set<Object> seen = new java.util.HashSet<Object>();
	for (Object o : array) {
		if (seen.contains(o))
			return true;
	}
	return false;
}
```
To fix this problem, a statement `seen.add(o);` should be added to the end of the loop body to ensure that `seen` is correctly maintained.


## References
* Java API Specification: [Collection](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/util/Collection.html), [Map](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/util/Map.html).
* Common Weakness Enumeration: [CWE-561](https://cwe.mitre.org/data/definitions/561.html).
