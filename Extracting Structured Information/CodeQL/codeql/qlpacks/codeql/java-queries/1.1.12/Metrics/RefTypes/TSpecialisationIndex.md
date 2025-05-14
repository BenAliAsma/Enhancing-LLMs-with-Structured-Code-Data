# Type specialization index
Specialization index is the extent to which a subclass overrides the behavior of its ancestor classes. It is computed as follows:

1. Determine the number of overridden methods in the subclass (not counting overrides of abstract methods).
1. Multiply this number by the subclass's depth in the inheritance hierarchy.
1. Divide the result by the subclass's total number of methods.
If a class overrides many of the methods of its ancestor classes, it indicates that the abstractions in the ancestor classes should be reviewed. This is particularly true for subclasses that are lower down in the inheritance hierarchy. In general, subclasses should *add* behavior to their superclasses, rather than *redefining* the behavior that is already there.


## Recommendation
The most common reason that classes have a high specialization index is that multiple subclasses specialize a common base class in the same way. In this case, the relevant method(s) should be pulled up into the base class (see the 'Pull Up Method' refactoring in \[Fowler\]).


## Example
In the following example, duplicating `getName` in each of the subclasses is unnecessary.


```java
abstract class Animal {
    protected String animalName;

    public Animal(String name) {
        animalName = name;
    }

    public String getName(){
        return animalName;
    }
    public abstract String getKind();
}

class Dog extends Animal {
    public Dog(String name) {
        super(name);
    }

    public String getName() {  // This method duplicates the method in class 'Cat'.
        return animalName + " the " +  getKind();
    }

    public String getKind() {
        return "dog";
    }
 }

class Cat extends Animal {
    public Cat(String name) {
        super(name);
    }

    public String getName() {  // This method duplicates the method in class 'Dog'.
        return animalName + " the " +  getKind();
    }

    public String getKind() {
        return "cat";
    }
}

```
To decrease the specialization index of the subclasses, pull up `getName` into the base class.


```java
abstract class Animal {
    private String animalName;

    public Animal(String name) {
        animalName = name;
    }

    public String getName() {  // Method has been pulled up into the base class.
        return animalName + " the " +  getKind();
    }

    public abstract String getKind();
}

class Dog extends Animal {
    public Dog(String name) {
        super(name);
    }

    public String getKind() {
        return "dog";
    }
}

class Cat extends Animal {
    public Cat(String name) {
        super(name);
    }

    public String getKind() {
        return "cat";
    }
}

```

## References
* M. Fowler, *Refactoring*, pp. 260-3. Addison-Wesley, 1999.
* M. Lorenz and J. Kidd, *Object-oriented Software Metrics*. Prentice Hall, 1994.
* O. de Moor et al, *Keynote Address: .QL for Source Code Analysis*. Proceedings of the 7th IEEE International Working Conference on Source Code Analysis and Manipulation, 2007.
