# Use of default toString()
In most cases, calling the default implementation of `toString` in `java.lang.Object` is not what is intended when a string representation of an object is required. The output of the default `toString` method consists of the class name of the object as well as the object's hashcode, which is usually not what was intended.

This rule includes explicit and implicit calls to `toString` that resolve to `java.lang.Object.toString`, particularly calls that are used in print or log statements.


## Recommendation
For objects that are printed, define a `toString` method for the object that returns a human-readable string.


## Example
The following example shows that printing an object makes an implicit call to `toString`. Because the class `WrongPerson` does not have a `toString` method, `Object.toString` is called instead, which returns the class name and the `wp` object's hashcode.


```java
// This class does not have a 'toString' method, so 'java.lang.Object.toString'
// is used when the class is converted to a string.
class WrongPerson {
	private String name;
	private Date birthDate; 
	
	public WrongPerson(String name, Date birthDate) {
		this.name =name;
		this.birthDate = birthDate;
	}
}

public static void main(String args[]) throws Exception {
	DateFormat dateFormatter = new SimpleDateFormat("yyyy-MM-dd");
	WrongPerson wp = new WrongPerson("Robert Van Winkle", dateFormatter.parse("1967-10-31"));

	// BAD: The following statement implicitly calls 'Object.toString', 
	// which returns something similar to:
	// WrongPerson@4383f74d
	System.out.println(wp);
}
```
In contrast, in the following modification of the example, the class `Person` does have a `toString` method, which returns a string containing the arguments that were passed when the object `p` was created.


```java
// This class does have a 'toString' method, which is used when the object is
// converted to a string.
class Person {
	private String name;
	private Date birthDate;
	
	public String toString() {
		DateFormat dateFormatter = new SimpleDateFormat("yyyy-MM-dd");
		return "(Name: " + name + ", Birthdate: " + dateFormatter.format(birthDate) + ")";
	}
	
	public Person(String name, Date birthDate) {
		this.name =name;
		this.birthDate = birthDate;
	}
}

public static void main(String args[]) throws Exception {
	DateFormat dateFormatter = new SimpleDateFormat("yyyy-MM-dd");
	Person p = new Person("Eric Arthur Blair", dateFormatter.parse("1903-06-25"));

	// GOOD: The following statement implicitly calls 'Person.toString', 
	// which correctly returns a human-readable string:
	// (Name: Eric Arthur Blair, Birthdate: 1903-06-25)
	System.out.println(p);
}
```

## References
* J. Bloch, *Effective Java (second edition)*, Item 10. Addison-Wesley, 2008.
* Java API Specification: [Object.toString()](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/Object.html#toString()).
