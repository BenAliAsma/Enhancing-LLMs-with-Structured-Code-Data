# Assignment in Boolean expression
The assignment operator (`=`) can easily be confused with the equality operator (`==`), and can make a Boolean expression more difficult to understand. Consequently, assignments in Boolean expressions should be avoided.

Some useful idioms are an exception to this rule, such as checking that some bytes have been read from an input-stream, as shown in the `readConfiguration` method in the example below. More precisely, an assignment is allowed in a Boolean expression if the result of the assignment is compared to another value.


## Recommendation
Consider structuring the condition so that the side-effects are moved outside of the condition, possibly splitting the condition into several separate tests.


## Example
In the following example, consider the rather confusing assignment to `restart` in the `notify` method. The assignment should be performed outside of the condition instead.


```java
public class ScreenView
{
    private static int BUF_SIZE = 1024;
    private Screen screen;

    public void notify(Change change) {
        boolean restart = false;
        if (change.equals(Change.MOVE)
            || v.equals(Change.REPAINT)
            || (restart = v.equals(Change.RESTART))  // AVOID: Confusing assignment in condition
            || v.equals(Change.FLIP))
        {
            if (restart)
                WindowManager.restart();
            screen.update();
        }
    }

    // ...

    public void readConfiguration(InputStream config) {
        byte[] buf = new byte[BUF_SIZE];
        int read;
        while ((read = config.read(buf)) > 0) {  // OK: Assignment whose result is compared to
                                                 // another value
            // ...
        }
        // ...
    }
}

```

## References
* Help - Eclipse Platform: [Java Compiler Errors/Warnings Preferences](https://help.eclipse.org/2020-12/advanced/content.jsp?topic=/org.eclipse.jdt.doc.user/reference/preferences/java/compiler/ref-preferences-errors-warnings.htm).
* Java Language Specification: [15.21 Equality Operators](https://docs.oracle.com/javase/specs/jls/se11/html/jls-15.html#jls-15.21), [15.26 Assignment Operators](https://docs.oracle.com/javase/specs/jls/se11/html/jls-15.html#jls-15.26).
* Common Weakness Enumeration: [CWE-481](https://cwe.mitre.org/data/definitions/481.html).
