# notify instead of notifyAll
Calls to the `notify` method rather than `notifyAll` may fail to wake up the correct thread if an object's monitor (intrinsic lock) is used for multiple conditions. `notify` only wakes up a single arbitrary thread that is waiting on the object's monitor, whereas `notifyAll` wakes up all such threads.


## Recommendation
Ensure that the call to `notify` instead of `notifyAll` is a correct and desirable optimization. If not, call `notifyAll` instead.


## Example
In the following example, the methods `produce` and `consume` both use `notify` to tell any waiting threads that an object has been added or removed from the buffer. However, this means that only *one* thread is notified. The woken-up thread might not be able to proceed due to its condition being false, immediately going back to the waiting state. As a result no progress is made.


```java
class ProducerConsumer {
    private static final int MAX_SIZE=3;
    private List<Object> buf = new ArrayList<Object>();

    public synchronized void produce(Object o) {
        while (buf.size()==MAX_SIZE) {
            try {
                wait();
            }
            catch (InterruptedException e) {
               ...
            }
        }
        buf.add(o);
        notify(); // 'notify' is used
    }

    public synchronized Object consume() {

        while (buf.size()==0) {
            try {
                wait();
            }
            catch (InterruptedException e) {
                ...
            }
        }
        Object o = buf.remove(0);
        notify(); // 'notify' is used
        return o;
    }
}

```
When using `notifyAll` instead of `notify`, *all* threads are notified, and if there are any threads that could proceed, we can be sure that at least one of them will do so.


## References
* J. Bloch. *Effective Java (second edition)*, p. 277. Addison-Wesley, 2008.
* Java API Specification: [Object.notify()](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/Object.html#notify()), [Object.notifyAll()](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/Object.html#notifyAll()).
* Common Weakness Enumeration: [CWE-662](https://cwe.mitre.org/data/definitions/662.html).
