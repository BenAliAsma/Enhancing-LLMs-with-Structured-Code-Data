# Wait outside loop
Calling `Object.wait` outside of a loop may cause problems because the thread does not go back to sleep after a spurious wake-up call. This results in the program continuing before the expected condition is met.


## Recommendation
Ensure that `wait` is called within a loop that tests for the condition that the thread is waiting for. This ensures that the program only proceeds to execute when the relevant condition is true. Note that the thread that calls `wait` on an object must be the owner of that object's monitor.


## Example
In the following example, `obj.wait` is called within a `while` loop until the condition is true, at which point the program continues with the next statement after the loop:


```java
synchronized (obj) {
    while (<condition is false>) obj.wait();
    // condition is true, perform appropriate action ...
}
```

## References
* J. Bloch, *Effective Java (second edition)*, p. 276. Addison-Wesley, 2008.
* Java API Specification: [Object.wait()](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/Object.html#wait()).
* The Java Tutorials: [Guarded Blocks](https://docs.oracle.com/javase/tutorial/essential/concurrency/guardmeth.html).
