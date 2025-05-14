# Empty body of finalizer
An empty `finalize` method is useless and may prevent finalization from working properly. This is because, unlike a constructor, a finalizer does not implicitly call the finalizer of the superclass. Thus, an empty finalizer prevents any finalization logic that is defined in any of its superclasses from being executed.


## Recommendation
Do not include an empty `finalize` method.


## Example
In the following example, the empty `finalize` method in class `ExtendedLog` prevents the `finalize` method in class `Log` from being called. The result is that the log file is not closed. To fix this, remove the empty `finalize` method.


```java
class ExtendedLog extends Log
{
	// ...

	protected void finalize() {
		// BAD: This empty 'finalize' stops 'super.finalize' from being executed.
	}
}

class Log implements Closeable
{
	// ...

	public void close() {
		// ...
	}

	protected void finalize() {
		close();
	}
}
```

## References
* Java Language Specification: [12.6 Finalization of Class Instances](https://docs.oracle.com/javase/specs/jls/se11/html/jls-12.html#jls-12.6).
* Common Weakness Enumeration: [CWE-568](https://cwe.mitre.org/data/definitions/568.html).
