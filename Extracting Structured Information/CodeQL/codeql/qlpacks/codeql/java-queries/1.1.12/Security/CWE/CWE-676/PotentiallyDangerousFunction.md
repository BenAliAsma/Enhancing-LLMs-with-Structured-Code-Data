# Use of a potentially dangerous function
This rule finds calls to methods that are dangerous to use. Currently, it checks for calls to `Thread.stop`.

Stopping a thread with `Thread.stop` causes it to receive a `ThreadDeath` exception. That exception propagates up the stack, releasing all monitors that the thread was holding. In some cases the relevant code will be protected by catching the `ThreadDeath` exception and cleaning up, but because the exception can potentially be thrown from so very many locations, it is impractical to catch all such cases. As a result, calling `Thread.stop` is likely to result in corrupt data.


## Recommendation
The best solution is usually to provide an alternate communication mechanism for the thread that might need to be interrupted early. For example, Oracle gives the following example of using a volatile variable to communicate whether the worker thread should exit:


```java
private volatile Thread blinker;

public void stop() {
    blinker = null;
}

public void run() {
    Thread thisThread = Thread.currentThread();
    while (blinker == thisThread) {
        try {
            Thread.sleep(interval);
        } catch (InterruptedException e){
        }
        repaint();
    }
}

```
It is also possible to use `Thread.interrupt` and to catch and handle `InterruptedException` when it occurs. However, it can be difficult to handle an `InterruptedException` everywhere it might occur; for example, the sample code above simply discards the exception rather than actually exiting the thread.

Another strategy is to use message passing, for example via a `BlockingQueue`. In addition to passing the worker thread its ordinary work via such a message queue, the worker can be asked to exit by a particular kind of message being sent on the queue.


## References
* SEI CERT Oracle Coding Standard for Java: [THI05-J. Do not use Thread.stop() to terminate threads](https://wiki.sei.cmu.edu/confluence/display/java/THI05-J.+Do+not+use+Thread.stop()+to+terminate+threads).
* Java API Specification: [Java Thread Primitive Deprecation](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/doc-files/threadPrimitiveDeprecation.html).
* Java API Specification: [Thread.interrupt](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/Thread.html#interrupt()), [BlockingQueue](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/util/concurrent/BlockingQueue.html).
* Common Weakness Enumeration: [CWE-676](https://cwe.mitre.org/data/definitions/676.html).
