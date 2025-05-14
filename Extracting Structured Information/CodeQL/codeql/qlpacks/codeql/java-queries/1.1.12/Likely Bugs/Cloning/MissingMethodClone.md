# No clone method
A class that implements `Cloneable` should override `Object.clone`. For non-trivial objects, the `Cloneable` contract requires a deep copy of the object's internal state. A class that does not have a `clone` method indicates that the class is breaking the contract and will have undesired behavior.

The Java API Specification states that, for an object `x`, the general intent of the `clone` method is for it to satisfy the following three properties:

* `x.clone() != x` (the cloned object is a different object instance)
* `x.clone().getClass() == x.getClass()` (the cloned object is the same type as the source object)
* `x.clone().equals(x)` (the cloned object has the same 'contents' as the source object)
For the cloned object to be of the same type as the source object, non-final classes must call `super.clone` and that call must eventually reach `Object.clone`, which creates an instance of the right type. If it were to create a new object using a constructor, a subclass that does not implement the `clone` method returns an object of the wrong type. In addition, all of the class's supertypes that also override `clone` must call `super.clone`. Otherwise, it never reaches `Object.clone` and creates an object of the incorrect type.

However, as `Object.clone` only does a shallow copy of the fields of an object, any `Cloneable` objects that have a "deep structure" (for example, objects that use an array or `Collection`) must take the clone that results from the call to `super.clone` and assign explicitly created copies of the structure to the clone's fields. This means that the cloned instance does not share its internal state with the source object. If it *did* share its internal state, any changes made in the cloned object would also affect the internal state of the source object, probably causing unintended behavior.

One added complication is that `clone` cannot modify values in final fields, which would be already set by the call to `super.clone`. Some fields must be made non-final to correctly implement the `clone` method.


## Recommendation
The necessity of creating a deep copy of an object's internal state means that, for most objects, `clone` must be overridden to satisfy the `Cloneable` contract. Implement a `clone` method that properly creates the internal state of the cloned object.

Notable exceptions to this recommendation are:

* Classes that contain only primitive types (which will be properly cloned by `Object.clone` as long as its `Cloneable` supertypes all call `super.clone`).
* Subclasses of `Cloneable` classes that do not introduce new state.

## Example
In the following example, `WrongStack` does not implement `clone`. This means that when `ws1clone` is cloned from `ws1`, the default `clone` implementation is used. This results in operations on the `ws1clone` stack affecting the `ws1` stack.


```java
abstract class AbstractStack implements Cloneable {
    public AbstractStack clone() {
        try {
            return (AbstractStack) super.clone();
        } catch (CloneNotSupportedException e) {
            throw new AssertionError("Should not happen");
        }
    }
}

class WrongStack extends AbstractStack {
    private static final int MAX_STACK = 10;
    int[] elements = new int[MAX_STACK];
    int top = -1;

    void push(int newInt) {
        elements[++top] = newInt;
    }
    int pop() {
        return elements[top--];
    }
    // BAD: No 'clone' method to create a copy of the elements.
    // Therefore, the default 'clone' implementation (shallow copy) is used, which
    // is equivalent to:
    //
    //  public WrongStack clone() {
    //      WrongStack cloned = (WrongStack) super.clone();
    //      cloned.elements = elements;  // Both 'this' and 'cloned' now use the same elements.
    //      return cloned;
    //  }
}

public class MissingMethodClone {
    public static void main(String[] args) {
        WrongStack ws1 = new WrongStack();              // ws1: {}
        ws1.push(1);                                    // ws1: {1}
        ws1.push(2);                                    // ws1: {1,2}
        WrongStack ws1clone = (WrongStack) ws1.clone(); // ws1clone: {1,2}
        ws1clone.pop();                                 // ws1clone: {1}
        ws1clone.push(3);                               // ws1clone: {1,3}
        System.out.println(ws1.pop());                  // Because ws1 and ws1clone have the same
                                                        // elements, this prints 3 instead of 2
    }
}



```
In the following modified example, `RightStack` *does* implement `clone`. This means that when `rs1clone` is cloned from `rs1`, operations on the `rs1clone` stack do not affect the `rs1` stack.


```java
abstract class AbstractStack implements Cloneable {
    public AbstractStack clone() {
        try {
            return (AbstractStack) super.clone();
        } catch (CloneNotSupportedException e) {
            throw new AssertionError("Should not happen");
        }
    }
}

class RightStack extends AbstractStack {
    private static final int MAX_STACK = 10;
    int[] elements = new int[MAX_STACK];
    int top = -1;

    void push(int newInt) {
        elements[++top] = newInt;
    }
    int pop() {
        return elements[top--];
    }

    // GOOD: 'clone' method to create a copy of the elements.
    public RightStack clone() {
        RightStack cloned = (RightStack) super.clone();
        cloned.elements = elements.clone();  // 'cloned' has its own elements.
        return cloned;
    }
}

public class MissingMethodClone {
    public static void main(String[] args) {
        RightStack rs1 = new RightStack();              // rs1: {}
        rs1.push(1);                                    // rs1: {1}
        rs1.push(2);                                    // rs1: {1,2}
        RightStack rs1clone = rs1.clone();              // rs1clone: {1,2}
        rs1clone.pop();                                 // rs1clone: {1}
        rs1clone.push(3);                               // rs1clone: {1,3}
        System.out.println(rs1.pop());                  // Correctly prints 2
    }
}



```

## References
* J. Bloch, *Effective Java (second edition)*, Item 11. Addison-Wesley, 2008.
* Java API Specification: [Object.clone()](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/Object.html#clone()).
