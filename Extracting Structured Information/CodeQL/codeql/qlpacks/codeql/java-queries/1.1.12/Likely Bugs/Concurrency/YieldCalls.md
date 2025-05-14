# Call to Thread.yield()
The method `Thread.yield` is a non-portable and underspecified operation. It may have no effect, and is not a reliable way to prevent a thread from taking up too much execution time.


## Recommendation
Use alternative ways of preventing a thread from taking up too much execution time. Communication between threads should normally be implemented using some form of waiting for a notification using the `wait` and `notifyAll` methods or by using the `java.util.concurrent` library.

In some cases, calls to `Thread.sleep` may be appropriate to temporarily cease execution (provided there is no possibility for race conditions), but this is not generally recommended.


## References
* J. Bloch, *Effective Java (second edition)*, Item 72. Addison-Wesley, 2008.
* Java API Specification: [Thread.yield()](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/Thread.html#yield()), [Object.wait()](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/Object.html#wait()), [Object.notifyAll()](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/Object.html#notifyAll()), [java.util.concurrent](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/util/concurrent/package-summary.html), [Thread.sleep()](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/Thread.html#sleep(long)).
