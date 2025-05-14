# Wait with two locks held
Calling `Object.wait` while two locks are held may cause deadlock, because only one lock is released by `wait`.


## Recommendation
See if one of the locks should continue to be held while waiting for a condition on the other lock. If not, release one of the locks before calling `Object.wait`.


## Example
In the following example of the problem, `printText` locks both `idLock` and `textLock` before it reads the value of `text`. It then calls `textLock.wait`, which releases the lock on `textLock`. However, `setText` needs to lock `idLock` but it cannot because `idLock` is still locked by `printText`. Thus, deadlock is caused.


```java
class WaitWithTwoLocks {

    private final Object idLock = new Object();
    private int id = 0;

    private final Object textLock = new Object();
    private String text = null;

    public void printText() {
        synchronized (idLock) {
            synchronized (textLock) {
                while(text == null)
                    try {
                        textLock.wait();  // The lock on "textLock" is released but not the
                                          // lock on "idLock".
                    }
                    catch (InterruptedException e) { ... }
                System.out.println(id + ":" + text);
                text = null;
                textLock.notifyAll();
            }
        }
    }

    public void setText(String mesg) {
        synchronized (idLock) { // "setText" needs a lock on "idLock" but "printText" already
                                // holds a lock on "idLock", leading to deadlock
            synchronized (textLock) {
                id++;
                text = mesg;
                idLock.notifyAll();
                textLock.notifyAll();
            }
        }
    }
 }

```
In the following modification of the above example, `id` and `text` are included in the class `Message`. The method `printText` synchronizes on the field `message` before it reads the value of `message.text`. It then calls `message.wait`, which releases the lock on `message`. This enables `setText` to lock `message` so that it can proceed.


```java
class WaitWithTwoLocksGood {

    private static class Message {
        public int id = 0;
        public String text = null;
    }

    private final Message message = new Message();

    public void printText() {
        synchronized (message) {
            while(message.txt == null)
                try {
                    message.wait();
                }
                catch (InterruptedException e) { ... }
            System.out.println(message.id + ":" + message.text);
            message.text = null;
            message.notifyAll();
        }
    }

    public void setText(String mesg) {
        synchronized (message) {
            message.id++;
            message.text = mesg;
            message.notifyAll();
        }
    }
 }

```

## References
* Java API Specification: [Object.wait()](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/Object.html#wait()).
* Common Weakness Enumeration: [CWE-833](https://cwe.mitre.org/data/definitions/833.html).
