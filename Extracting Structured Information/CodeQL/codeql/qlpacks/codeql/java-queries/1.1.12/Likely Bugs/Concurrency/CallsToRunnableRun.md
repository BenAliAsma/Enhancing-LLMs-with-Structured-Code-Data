# Direct call to a run() method
A direct call of a `Thread` object's `run` method does not start a separate thread. The method is executed within the current thread. This is an unusual use because `Thread.run()` is normally intended to be called from within a separate thread.


## Recommendation
To execute `Runnable.run` from within a separate thread, do one of the following:

* Construct a `Thread` object using the `Runnable` object, and call `start` on the `Thread` object.
* Define a subclass of a `Thread` object, and override the definition of its `run` method. Then construct an instance of this subclass and call `start` on that instance directly.

## Example
In the following example, the main thread, `ThreadDemo`, calls the child thread, `NewThread`, using `run`. This causes the child thread to run to completion before the rest of the main thread is executed, so that "Child thread activity" is printed before "Main thread activity".


```java
public class ThreadDemo {
    public static void main(String args[]) {
        NewThread runnable = new NewThread();

        runnable.run();    // Call to 'run' does not start a separate thread

        System.out.println("Main thread activity.");
    }
}

class NewThread extends Thread {
    public void run() {
        try {
            Thread.sleep(10000);
        }
        catch (InterruptedException e) {
            System.out.println("Child interrupted.");
        }
        System.out.println("Child thread activity.");
    }
}
```
To enable the two threads to run concurrently, create the child thread and call `start`, as shown below. This causes the main thread to continue while the child thread is waiting, so that "Main thread activity" is printed before "Child thread activity".


```java
public class ThreadDemo {
    public static void main(String args[]) {
    	NewThread runnable = new NewThread();
    	
        runnable.start();                                         // Call 'start' method
        
        System.out.println("Main thread activity.");
    }
}
```

## References
* The Java Tutorials: [Defining and Starting a Thread](https://docs.oracle.com/javase/tutorial/essential/concurrency/runthread.html).
* Common Weakness Enumeration: [CWE-572](https://cwe.mitre.org/data/definitions/572.html).
