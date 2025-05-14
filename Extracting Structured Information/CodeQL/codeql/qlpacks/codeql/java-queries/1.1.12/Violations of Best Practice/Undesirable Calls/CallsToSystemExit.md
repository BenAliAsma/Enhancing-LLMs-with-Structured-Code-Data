# Forcible JVM termination
Calling one of the methods `System.exit`, `Runtime.halt`, and `Runtime.exit` immediately terminates the Java Virtual Machine (JVM), effectively killing all threads without giving any of them a chance to perform cleanup actions or recover. As such, it is a dangerous thing to do: firstly, it can terminate the entire program inadvertently, and secondly, it can prevent important resources from being released or program state from being written to disk consistently.

It is sometimes considered acceptable to call `System.exit` from a program's `main` method in order to indicate the overall exit status of the program. Such calls are an exception to this rule.


## Recommendation
It is usually preferable to use a different mechanism for reporting failure conditions. Consider returning a special value (perhaps `null`) that users of the current method check for and recover from appropriately. Alternatively, throw a suitable exception, which unwinds the stack and allows properly written code to clean up after itself, while leaving other threads undisturbed.


## Example
In the following example, problem 1 shows that `FileOutput.write` tries to write some data to disk and terminates the JVM if this fails. This leaves the partially-written file on disk without any cleanup code running. It would be better to either return `false` to indicate the failure, or let the `IOException` propagate upwards and be handled by a method that knows how to recover.

Problem 2 is more subtle. In this example, there is just one entry point to the program (the `main` method), which constructs an `Action` and performs it. `Action.run` calls `System.exit` to indicate successful completion. Consider, however, how this code might be integrated in an application server that constructs `Action` instances and calls `run` on them without going through `main`. The fact that `run` terminates the JVM instead of returning its exit code as an integer makes that use-case impossible.


```java
// Problem 1: Miss out cleanup code 
class FileOutput {
    boolean write(String[] s) {
        try {
            output.write(s.getBytes());
        } catch (IOException e) {
            System.exit(1);
        }
        return true;
    }
}

// Problem 2: Make code reuse difficult
class Action {
    public void run() {
        // ...
        // Perform tasks ...
        // ...
        System.exit(0);
    }
    public static void main(String[] args) {
        new Action(args).run();
    }
}
```

## References
* J. Bloch, *Effective Java (second edition)*, p. 232. Addison-Wesley, 2008.
* Java API Specification: [System.exit(int)](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/System.html#exit(int)), [Runtime.halt(int)](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/Runtime.html#halt(int)), [Runtime.exit(int)](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/Runtime.html#exit(int)).
* Common Weakness Enumeration: [CWE-382](https://cwe.mitre.org/data/definitions/382.html).
