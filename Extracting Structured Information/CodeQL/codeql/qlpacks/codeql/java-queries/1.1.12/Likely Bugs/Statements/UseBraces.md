# Misleading indentation
A control structure (an `if` statement or a loop) has a body that is either a block of statements surrounded by curly braces or a single statement.

If you omit braces, it is particularly important to ensure that the indentation of the code matches the control flow of the code.


## Recommendation
It is usually considered good practice to include braces for all control structures in Java. This is because it makes it easier to maintain the code later. For example, it's easy to see at a glance which part of the code is in the scope of an `if` statement, and adding more statements to the body of the `if` statement is less error-prone.

You should also ensure that the indentation of the code is consistent with the actual flow of control, so that it does not confuse programmers.


## Example
In the example below, the original version of `Cart` is missing braces. This means that the code triggers a `NullPointerException` at runtime if `i` is `null`.


```java
class Cart {
    Map<Integer, Integer> items = ...
    public void addItem(Item i) {
        // No braces and misleading indentation.
        if (i != null)
            log("Adding item: " + i);
            // Indentation suggests that the following statements
            // are in the body of the 'if'.
            Integer curQuantity = items.get(i.getID());
            if (curQuantity == null) curQuantity = 0;
            items.put(i.getID(), curQuantity+1);
    }
}
```
The corrected version of `Cart` does include braces, so that the code executes as the indentation suggests.


```java
class Cart {
    Map<Integer, Integer> items = ...
    public void addItem(Item i) {
        // Braces included.
        if (i != null) {
            log("Adding item: " + i);
            Integer curQuantity = items.get(i.getID());
            if (curQuantity == null) curQuantity = 0;
            items.put(i.getID(), curQuantity+1);
        }
    }
}
```
In the following example the indentation may or may not be misleading depending on your tab width settings. As such, mixing tabs and spaces in this way is not recommended, since what looks fine in one context can be very misleading in another.


```java
// Tab width 8
        if (b)       // Indentation: 1 tab
                f(); // Indentation: 2 tabs
        g();         // Indentation: 8 spaces

// Tab width 4
    if (b)   // Indentation: 1 tab
        f(); // Indentation: 2 tabs
        g(); // Indentation: 8 spaces
```
If you mix tabs and spaces in this way, then you might get seemingly false positives, since your tab width settings cannot be taken into account.


## References
* Java SE Documentation: [Compound Statements](https://www.oracle.com/java/technologies/javase/codeconventions-statements.html#15395).
* Wikipedia: [Indentation style](https://en.wikipedia.org/wiki/Indentation_style).
