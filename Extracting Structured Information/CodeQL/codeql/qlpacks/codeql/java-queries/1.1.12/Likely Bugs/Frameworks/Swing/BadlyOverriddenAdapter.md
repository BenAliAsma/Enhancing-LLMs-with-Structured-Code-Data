# Bad implementation of an event Adapter
Event adapters in Swing (and Abstract Window Toolkit) provide a convenient way for programmers to implement event listeners. However, care must be taken to get the names of the overridden methods right, or the event handlers will not be called.


## In Depth
The event listener interfaces in Swing (and Abstract Window Toolkit) have many methods. For example, `java.awt.event.MouseListener` is defined as follows:

```java
public interface MouseListener extends EventListener {
    public abstract void mouseClicked(MouseEvent);
    public abstract void mousePressed(MouseEvent);
    public abstract void mouseReleased(MouseEvent);
    public abstract void mouseEntered(MouseEvent);
    public abstract void mouseExited(MouseEvent);
}
```
The large number of methods can make such interfaces lengthy and tedious to implement, especially because it is rare that all of the methods need to be overridden. It is much more common that you need to override only one method, for example the `mouseClicked` event.

For this reason, Swing supplies *adapter* classes that provide default, blank implementations of interface methods. An example is `MouseAdapter`, which provides default implementations for the methods in `MouseListener`, `MouseWheelListener` and `MouseMotionListener`. (Note that an adapter often implements multiple interfaces to avoid a large number of small adapter classes.) This makes it easy for programmers to implement just the methods they need from a given interface.

Unfortunately, adapter classes are also a source of potential defects. Because the `@Override` annotation is not compulsory, it is very easy for programmers not to use it and then mistype the name of the method. This introduces a new method rather than implementing the relevant event handler.


## Recommendation
Ensure that any overriding methods have exactly the same name as the overridden method.


## Example
In the following example, the programmer has tried to implement the `mouseClicked` function but has misspelled the function name. This makes the function inoperable but the programmer gets no warning about this from the compiler.


```java
add(new MouseAdapter() {
    public void mouseClickd(MouseEvent e) {
        // ...
    }
});
```
In the following modified example, the function name is spelled correctly. It is also preceded by the `@Override` annotation, which will cause the compiler to display an error if there is not a function of the same name to be overridden.


```java
add(new MouseAdapter() {
    @Override
    public void mouseClicked(MouseEvent e) {
        // ...
    }
});
```

## References
* D. Flanagan, *Java Foundation Classes in a Nutshell*, Chapter 26. O'Reilly, 1999.
* Java API Specification: [Annotation Type Override](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/Override.html).
* The Java Tutorials: [Event Adapters](https://docs.oracle.com/javase/tutorial/uiswing/events/generalrules.html#eventAdapters).
