# Unreleased lock
When a thread acquires a lock it must make sure to unlock it again; failing to do so can lead to deadlocks. If a lock allows a thread to acquire it multiple times, for example `java.util.concurrent.locks.ReentrantLock`, then the number of locks must match the number of unlocks in order to fully release the lock.


## Recommendation
It is recommended practice always to immediately follow a call to `lock` with a `try` block and place the call to `unlock` inside the `finally` block. Beware of calls inside the `finally` block that could cause exceptions, as this may result in skipping the call to `unlock`.


## Example
The typical pattern for using locks safely looks like this:


```java
public void m() {
   lock.lock();
   // A
   try {
      // ... method body
   } finally {
      // B
      lock.unlock();
   }
}
```
If any code that can cause a premature method exit (for example by throwing an exception) is inserted at either point `A` or `B` then the method might not unlock, so this should be avoided.


## References
* Java API Specification: [java.util.concurrent.locks.Lock](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/util/concurrent/locks/Lock.html), [java.util.concurrent.locks.ReentrantLock](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/util/concurrent/locks/ReentrantLock.html).
* Common Weakness Enumeration: [CWE-764](https://cwe.mitre.org/data/definitions/764.html).
* Common Weakness Enumeration: [CWE-833](https://cwe.mitre.org/data/definitions/833.html).
