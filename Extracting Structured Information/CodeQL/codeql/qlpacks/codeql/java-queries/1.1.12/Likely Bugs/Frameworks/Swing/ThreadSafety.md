# Thread safety
Because Swing components are not thread-safe (that is, they do not support concurrent access from multiple threads), Swing has a rule that states that method calls on Swing components that have been *realized* (see below) must be made from a special thread known as the *event-dispatching thread*. Failure to observe this rule may result in multiple threads accessing a Swing component concurrently, with the potential for deadlocks, race conditions and other errors related to multi-threading.

A component is considered *realized* if its `paint` method has been, or could be, called at this point. Realization is triggered according to the following rules:

* A top-level window is realized if `setVisible(true)`, `show` or `pack` is called on it.
* Realizing a container realizes the components it contains.
There are a few exceptions to the rule. These are documented more fully in \[The Swing Connection\] but the key exceptions are:

* It is safe to call the `repaint`, `revalidate` and `invalidate` methods on a Swing component from any thread.
* It is safe to add and remove listeners from any thread. Therefore, any method of the form `add*Listener` or `remove*Listener` is thread-safe.

## Recommendation
Ensure that method calls on Swing components are made from the event-dispatching thread. If you need to call a method on a Swing component from another thread, you must do so using the event-dispatching thread. Swing provides a `SwingUtilities` class that you can use to ask the event-dispatching thread to run arbitrary code on your components, by calling one of two methods. Each method takes a `Runnable` as its only argument:

* `SwingUtilities.invokeLater` asks the event-dispatching thread to run some code and then immediately returns (that is, it is non-blocking). The code is run at some indeterminate time in the future, but the thread that calls `invokeLater` does not wait for it.
* `SwingUtilities.invokeAndWait` asks the event-dispatching thread to run some code and then waits for it to complete (that is, it is blocking).

## Example
In the following example, there is a call from the main thread to a method, `setTitle`, on the `MyFrame` object after the object has been realized by the `setVisible(true)` call. This represents an unsafe call to a Swing method from a thread other than the event-dispatching thread.


```java
class MyFrame extends JFrame {
    public MyFrame() {
        setSize(640, 480);
        setTitle("BrokenSwing");
    }
}

public class BrokenSwing {
    private static void doStuff(MyFrame frame) {
        // BAD: Direct call to a Swing component after it has been realized
        frame.setTitle("Title");
    }

    public static void main(String[] args) {
        MyFrame frame = new MyFrame();
        frame.setVisible(true);
        doStuff(frame);
    }
}
```
In the following modified example, the call to `setTitle` is instead called from within a call to `invokeLater`.


```java
class MyFrame extends JFrame {
    public MyFrame() {
        setSize(640, 480);
        setTitle("BrokenSwing");
    }
}

public class GoodSwing {
    private static void doStuff(final MyFrame frame) {
        SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                // GOOD: Call to Swing component made via the
                // event-dispatching thread using 'invokeLater'
                frame.setTitle("Title");
            }
        });
    }

    public static void main(String[] args) {
        MyFrame frame = new MyFrame();
        frame.setVisible(true);
        doStuff(frame);
    }
}

```

## References
* D. Flanagan, *Java Foundation Classes in a Nutshell*, p.28. O'Reilly, 1999.
* Java Developer's Journal: [Building Thread-Safe GUIs with Swing](http://www.comscigate.com/JDJ/archives/0605/ford/index.html).
* The Java Tutorials: [Concurrency in Swing](https://docs.oracle.com/javase/tutorial/uiswing/concurrency/index.html).
* The Swing Connection: [Threads and Swing](https://www.comp.nus.edu.sg/~cs3283/ftp/Java/swingConnect/archive/tech_topics_arch/threads/threads.html).
