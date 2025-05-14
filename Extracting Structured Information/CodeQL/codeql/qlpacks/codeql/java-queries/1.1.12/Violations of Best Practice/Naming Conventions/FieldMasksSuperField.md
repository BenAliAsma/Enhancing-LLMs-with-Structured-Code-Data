# Field masks field in super class
A field that has the same name as a field in a superclass *hides* the field in the superclass. Such hiding might be unintentional, especially if there are no references to the hidden field using the `super` qualifier. In any case, it makes code more difficult to read.


## Recommendation
Ensure that any hiding is intentional. For clarity, it may be better to rename the field in the subclass.


## Example
In the following example, the programmer unintentionally added an `age` field to `Employee`, which hides the `age` field in `Person`. The constructor in `Person` sets the `age ` field in `Person` to 20 but the `age` field in `Employee` is still 0. This means that the program outputs 0, which is probably not what was intended.


```java
public class FieldMasksSuperField {
    static class Person {
        protected int age;
        public Person(int age)
        {
            this.age = age;
        }
    }

    static class Employee extends Person {
        protected int age;  // This field hides 'Person.age'.
        protected int numberOfYearsEmployed;
        public Employee(int age, int numberOfYearsEmployed)
        {
            super(age);
            this.numberOfYearsEmployed = numberOfYearsEmployed;
        }
    }

    public static void main(String[] args) {
        Employee e = new Employee(20, 2);
        System.out.println(e.age);
    }
}
```
To fix this, delete the declaration of `age` on line 11.


## References
* Help - Eclipse Platform: [Java Compiler Errors/Warnings Preferences](https://help.eclipse.org/2020-12/advanced/content.jsp?topic=/org.eclipse.jdt.doc.user/reference/preferences/java/compiler/ref-preferences-errors-warnings.htm).
* The Java Tutorials: [Hiding Fields](https://docs.oracle.com/javase/tutorial/java/IandI/hidevariables.html).
