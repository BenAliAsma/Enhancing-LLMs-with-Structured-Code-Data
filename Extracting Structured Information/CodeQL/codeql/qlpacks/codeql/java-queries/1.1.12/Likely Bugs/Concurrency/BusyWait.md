# Busy wait
Trying to control thread interaction by periodically calling `Thread.sleep` within a loop while waiting for a condition to be satisfied is less effective than waiting for a notification. This is because the waiting thread may either sleep for an unnecessarily long time or wake up too frequently. This approach may also result in race conditions and, therefore, incorrect code.

Trying to control thread interaction by repeatedly checking a synchronized data structure without calling `Thread.sleep` or waiting for a notification may waste a lot of system resources and cause noticeable performance problems.


## Recommendation
See if communication between threads can be improved by using either of the following solutions:

* The `java.util.concurrent` library, preferably
* The `Object.wait` and `Object.notifyAll` methods
If following one of these recommendations is not feasible, ensure that race conditions cannot occur and precise timing is not required for program correctness.


## Example
In the following example, the `Receiver` thread sleeps for an unnecessarily long time (up to five seconds) until it has received the message.


```java
class Message {
    public String text = "";
}

class Receiver implements Runnable {
    private Message message;
    public Receiver(Message msg) {
        this.message = msg;
    }
    public void run() {
        while(message.text.isEmpty()) {
            try {
                Thread.sleep(5000);  // Sleep while waiting for condition to be satisfied
            } catch (InterruptedException e) { }
        }
        System.out.println("Message Received at " + (System.currentTimeMillis()/1000));
        System.out.println(message.text);
    }
}

class Sender implements Runnable {
    private Message message;
    public Sender(Message msg) {
        this.message = msg;
    }
    public void run() {
        System.out.println("Message sent at " + (System.currentTimeMillis()/1000));
        message.text = "Hello World";
    }
}

public class BusyWait {
    public static void main(String[] args) {
        Message msg = new Message();
        new Thread(new Receiver(msg)).start();
        new Thread(new Sender(msg)).start();
    }
}

```
In the following modification of the above example, the `Receiver` thread uses the recommended approach of waiting for a notification that the message has been sent. This means that the thread can respond immediately instead of sleeping.


```java
class Message {
    public String text = "";
}

class Receiver implements Runnable {
    private Message message;
    public Receiver(Message msg) {
        this.message = msg;
    }
    public void run() {
        synchronized(message) {
            while(message.text.isEmpty()) {
                try {
                    message.wait();  // Wait for a notification
                } catch (InterruptedException e) { }
            }
        }
        System.out.println("Message Received at " + (System.currentTimeMillis()/1000));
        System.out.println(message.text);
    }
}

class Sender implements Runnable {
    private Message message;
    public Sender(Message msg) {
        this.message = msg;
    }
    public void run() {
        System.out.println("Message sent at " + (System.currentTimeMillis()/1000));
        synchronized(message) {
            message.text = "Hello World";
            message.notifyAll();  // Send notification
        }
    }
}

public class BusyWait {
    public static void main(String[] args) {
        Message msg = new Message();
        new Thread(new Receiver(msg)).start();
        new Thread(new Sender(msg)).start();
    }
}

```

## References
* J. Bloch, *Effective Java (second edition)*, Item 72. Addison-Wesley, 2008.
* Java API Specification: [Object.wait()](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/Object.html#wait()), [Object.notifyAll()](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/Object.html#notifyAll()), [java.util.concurrent](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/util/concurrent/package-summary.html).
* The Java Tutorials: [Guarded Blocks](https://docs.oracle.com/javase/tutorial/essential/concurrency/guardmeth.html), [High Level Concurrency Objects](https://docs.oracle.com/javase/tutorial/essential/concurrency/highlevel.html).
