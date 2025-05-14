# Missing super clone
A `clone` method that is overridden in a subclass should call `super.clone`. Not doing so causes the subclass `clone` to return an object of the wrong type, which violates the contract for `Cloneable`.

The Java API Specification states that, for an object `x`, the general intent of the `clone` method is for it to satisfy the following three properties:

* `x.clone() != x` (the cloned object is a different object instance)
* `x.clone().getClass() == x.getClass()` (the cloned object is the same type as the source object)
* `x.clone().equals(x)` (the cloned object has the same 'contents' as the source object)
For the cloned object to be of the same type as the source object, non-final classes must call `super.clone` and that call must eventually reach `Object.clone`, which creates an instance of the right type. If it were to create a new object using a constructor, a subclass that does not implement the `clone` method returns an object of the wrong type. In addition, all of the class's supertypes that also override `clone` must call `super.clone`. Otherwise, it never reaches `Object.clone` and creates an object of the incorrect type.

However, as `Object.clone` only does a shallow copy of the fields of an object, any `Cloneable` objects that have a "deep structure" (for example, objects that use an array or `Collection`) must take the clone that results from the call to `super.clone` and assign explicitly created copies of the structure to the clone's fields. This means that the cloned instance does not share its internal state with the source object. If it *did* share its internal state, any changes made in the cloned object would also affect the internal state of the source object, probably causing unintended behavior.

One added complication is that `clone` cannot modify values in final fields, which would be already set by the call to `super.clone`. Some fields must be made non-final to correctly implement the `clone` method.


## Recommendation
Every clone method should always use `super.clone` to construct the cloned object. This ensures that the cloned object is ultimately constructed by `Object.clone`, which uses reflection to ensure that an object of the correct runtime type is created.


## Example
In the following example, the attempt to clone `WrongEmployee` fails because `super.clone` is implemented incorrectly in its superclass `WrongPerson`.


```java
class WrongPerson implements Cloneable {
    private String name;
    public WrongPerson(String name) { this.name = name; }
    // BAD: 'clone' does not call 'super.clone'.
    public WrongPerson clone() {
        return new WrongPerson(this.name);
    }
}

class WrongEmployee extends WrongPerson {
    public WrongEmployee(String name) {
        super(name);
    }
    // ALMOST RIGHT: 'clone' correctly calls 'super.clone',
    // but 'super.clone' is implemented incorrectly.
    public WrongEmployee clone() {
    	return (WrongEmployee)super.clone();
    }
}

public class MissingCallToSuperClone {
    public static void main(String[] args) {
        WrongEmployee e = new WrongEmployee("John Doe");
        WrongEmployee eclone = e.clone(); // Causes a ClassCastException
    }
}

```
However, in the following modified example, the attempt to clone `Employee` succeeds because `super.clone` is implemented correctly in its superclass `Person`.


```java
class Person implements Cloneable {
    private String name;
    public Person(String name) { this.name = name; }
    // GOOD: 'clone' correctly calls 'super.clone'
    public Person clone() {
        try {
            return (Person)super.clone();
        } catch (CloneNotSupportedException e) {
            throw new AssertionError("Should never happen");
        }
    }
}

class Employee extends Person {
    public Employee(String name) {
        super(name);
    }
    // GOOD: 'clone' correctly calls 'super.clone'
    public Employee clone() {
    	return (Employee)super.clone();
    }
}

public class MissingCallToSuperClone {
    public static void main(String[] args) {
        Employee e2 = new Employee("Jane Doe");
        Employee e2clone = e2.clone(); // 'clone' correctly returns an object of type 'Employee'
    }
}

```

## References
* J. Bloch, *Effective Java (second edition)*, Item 11. Addison-Wesley, 2008.
* Java API Specification: [Object.clone()](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/Object.html#clone()).
* Common Weakness Enumeration: [CWE-580](https://cwe.mitre.org/data/definitions/580.html).
