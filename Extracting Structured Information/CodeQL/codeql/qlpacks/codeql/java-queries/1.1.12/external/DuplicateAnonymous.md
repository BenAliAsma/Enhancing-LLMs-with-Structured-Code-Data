# Duplicate anonymous class
Anonymous classes are a common way of creating implementations of an interface or abstract class whose functionality is really only needed once. Duplicating the definition of an anonymous class in several places is usually a sign that refactoring is necessary.

Code duplication in general is highly undesirable for a range of reasons. The artificially inflated amount of code is more difficult to understand, and sequences of similar but subtly different lines can mask the real purpose or intention behind them. Also, there is always a risk that only one of several copies of the code is updated to address a defect or add a feature.


## Recommendation
Introduce a concrete class that contains the definition just once, and replace the anonymous classes with instances of this concrete class.


## Example
In the following example, the definition of the class `addActionListener` is duplicated for each button that needs to use it. A better solution is shown that defines just one class, `MultiplexingListener`, which is used by each button.


```java
// BAD: Duplicate anonymous classes:
button1.addActionListener(new ActionListener() {
    public void actionPerfored(ActionEvent e)
    {
        for (ActionListener listener: listeners)
            listeners.actionPerformed(e);
    }
});

button2.addActionListener(new ActionListener() {
    public void actionPerfored(ActionEvent e)
    {
        for (ActionListener listener: listeners)
            listeners.actionPerformed(e);
    }
});

// ... and so on.

// GOOD: Better solution:
class MultiplexingListener implements ActionListener {
    public void actionPerformed(ActionEvent e) {
        for (ActionListener listener : listeners)
            listener.actionPerformed(e);
    }
}

button1.addActionListener(new MultiplexingListener());
button2.addActionListener(new MultiplexingListener());
// ... and so on.

```

## References
* E. Juergens, F. Deissenboeck, B. Hummel, S. Wagner. *Do code clones matter?* Proceedings of the 31st International Conference on Software Engineering, 485-495, 2009.
