# Inconsistent synchronization for field
If a field is mostly accessed in a synchronized context, but occasionally accessed in a non-synchronized way, the non-synchronized accesses may lead to race conditions.


## Recommendation
Ensure that the non-synchronized field accesses are made synchronized, if required.


## Example
In the following example, `counter` is accessed in a synchronized way in *all but one* cases. If `modifyCounter` is called by a large number of threads that are running concurrently, the value of `counter` at the end of each call may not be zero. This is because the non-synchronized statement could be interleaved with updates to the counter that are performed by the other threads.


```java
class MultiThreadCounter {
    public int counter = 0;

    public void modifyCounter() {
        synchronized(this) {
            counter--;
        }
        synchronized(this) {
            counter--;
        }
        synchronized(this) {
            counter--;
        }
        counter = counter + 3;  // No synchronization
    }
}

```
To correct this, the last statement of `modifyCounter` should be enclosed in a `synchronized` statement.


## References
* Java Language Specification: [Synchronization](https://docs.oracle.com/javase/specs/jls/se11/html/jls-17.html#jls-17.1).
* Common Weakness Enumeration: [CWE-662](https://cwe.mitre.org/data/definitions/662.html).
