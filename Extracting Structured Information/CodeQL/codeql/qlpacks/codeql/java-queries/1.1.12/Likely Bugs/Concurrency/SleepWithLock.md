# Sleep with lock held
Calling `Thread.sleep` with a lock held may lead to very poor performance or even deadlock. This is because `Thread.sleep` does not cause a thread to release its locks.


## Recommendation
`Thread.sleep` should be called only outside of a `synchronized` block. However, a better way for threads to yield execution time to other threads may be to use either of the following solutions:

* The `java.util.concurrent` library
* The `wait` and `notifyAll` methods

## Example
In the following example of the problem, two threads, `StorageThread` and `OtherThread`, are started. Both threads output a message to show that they have started but then `StorageThread` locks `counter` and goes to sleep. The lock prevents `OtherThread` from locking `counter`, so it has to wait until `StorageThread` has woken up and unlocked `counter` before it can continue.


```java
class StorageThread implements Runnable{
    public static Integer counter = 0;
    private static final Object LOCK = new Object();

    public void run() {
        System.out.println("StorageThread started.");
        synchronized(LOCK) {  // "LOCK" is locked just before the thread goes to sleep
            try {
                Thread.sleep(5000);
            } catch (InterruptedException e) { ... }
        }
        System.out.println("StorageThread exited.");
    }
}

class OtherThread implements Runnable{
    public void run() {
        System.out.println("OtherThread started.");
        synchronized(StorageThread.LOCK) {
            StorageThread.counter++;
        }
        System.out.println("OtherThread exited.");
    }
}

public class SleepWithLock {
    public static void main(String[] args) {
        new Thread(new StorageThread()).start();
        new Thread(new OtherThread()).start();
    }
}

```
To avoid this problem, `StorageThread` should call `Thread.sleep` outside the `synchronized` block instead, so that `counter` is unlocked.


## References
* Java API Specification: [Thread.sleep()](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/Thread.html#sleep(long)), [Object.wait()](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/Object.html#wait()), [Object.notifyAll()](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/Object.html#notifyAll()), [java.util.concurrent](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/util/concurrent/package-summary.html).
* Common Weakness Enumeration: [CWE-833](https://cwe.mitre.org/data/definitions/833.html).
