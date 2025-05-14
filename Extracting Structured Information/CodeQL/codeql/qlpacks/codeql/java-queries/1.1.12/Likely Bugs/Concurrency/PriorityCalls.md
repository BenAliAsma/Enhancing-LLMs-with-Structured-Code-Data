# Explicit thread priority
Specifying thread priorities using calls to `Thread.setPriority` and `Thread.getPriority` is not portable and may have adverse consequences such as starvation.


## Recommendation
Avoid setting thread priorities to control interactions between threads. Using the default thread priority should be sufficient for most applications.

However, if you need to enforce a specific synchronization order, use one of the following alternatives:

* Waiting for a notification using the `wait` and `notifyAll` methods
* Using the `java.util.concurrent` library
In some cases, calls to `Thread.sleep` may be appropriate to temporarily stop execution (provided that there is no possibility for race conditions), but this is not generally recommended.


## References
* J. Bloch, *Effective Java (second edition)*, Item 72. Addison-Wesley, 2008.
* Inform IT: [Adding Multithreading Capability to Your Java Applications](http://www.informit.com/articles/article.aspx?p=26326&seqNum=5).
