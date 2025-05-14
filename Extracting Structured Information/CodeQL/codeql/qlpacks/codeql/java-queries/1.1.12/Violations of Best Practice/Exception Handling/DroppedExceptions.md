# Discarded exception
You should not drop an exception, because it indicates that an unusual program state has been reached. This usually requires corrective actions to be performed to recover from the exceptional state and try to resume normal program operation.


## Recommendation
You should do one of the following:

* Catch and handle the exception.
* Throw the exception to the outermost level of nesting.
Note that usually you should catch and handle a checked exception, but you can throw an unchecked exception to the outermost level.

There is occasionally a valid reason for ignoring an exception. In such cases, you should document the reason to improve the readability of the code. Alternatively, you can implement a static method with an empty body to handle these exceptions. Instead of dropping the exception altogether, you can then pass it to the static method with a string explaining the reason for ignoring it.


## Examples
The following example shows a dropped exception.


```java
// Dropped exception, with no information on whether 
// the exception is expected or not
synchronized void waitIfAutoSyncScheduled() {
	try {
		while (isAutoSyncScheduled) {
			this.wait(1000);
		}
	} catch (InterruptedException e) {
	}
}
```
The following example adds a comment to document why the exception may be ignored.


```java
synchronized void waitIfAutoSyncScheduled() {
	try {
		while (isAutoSyncScheduled) {
			this.wait(1000);
		}
	} catch (InterruptedException e) {
		// Expected exception. The file cannot be synchronized yet.
	}
}
```
The following example shows how you can improve code readability by defining a new utility method.


```java
// 'ignore' method. This method does nothing, but can be called
// to document the reason why the exception can be ignored.
public static void ignore(Throwable e, String message) {

}
```
The following example shows the exception being passed to `ignore` with a comment.


```java
// Exception is passed to 'ignore' method with a comment
synchronized void waitIfAutoSyncScheduled() {
	try {
		while (isAutoSyncScheduled) {
			this.wait(1000);
		}
	} catch (InterruptedException e) {
		Exceptions.ignore(e, "Expected exception. The file cannot be synchronized yet.");
	}
}
```

## References
* J. Bloch, *Effective Java (second edition)*, Item 65. Addison-Wesley, 2008.
* The Java Tutorials: [Unchecked Exceptions - The Controversy](https://docs.oracle.com/javase/tutorial/essential/exceptions/runtime.html).
* Common Weakness Enumeration: [CWE-391](https://cwe.mitre.org/data/definitions/391.html).
