# Field is never assigned a non-null value
It is good practice to initialize every field in a constructor explicitly. A field that is never assigned any value (except possibly `null`) just returns the default value when it is read, or throws a `NullPointerException`.


## Recommendation
A field whose value is always `null` (or the corresponding default value for primitive types, for example `0`) is not particularly useful. Ensure that the code contains an assignment or initialization for each field. To help satisfy this rule, it is good practice to explicitly initialize every field in the constructor, even if the default value is acceptable.

If the field is genuinely never expected to hold a non-default value, check the statements that read the field and ensure that they are not making incorrect assumptions about the value of the field. Consider completely removing the field and rewriting the statements that read it, as appropriate.


## Example
In the following example, the private field `name` is not initialized in the constructor (and thus is implicitly set to `null`), but there is a getter method to access it.


```java
class Person {
    private String name;
    private int age;

    public Person(String name, int age) {
        this.age = age;
    }

    public String getName() {
        return name;
    }

    public int getAge() {
        return age;
    }
}

```
Therefore, the following code throws a `NullPointerException`:

```java
Person p = new Person("Arthur Dent", 30);
int l = p.getName().length();
```
To fix the code, `name` should be initialized in the constructor by adding the following line: `this.name = name;`

