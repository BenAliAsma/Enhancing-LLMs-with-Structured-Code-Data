# Dead class
Classes that are never used at runtime are redundant and should be removed.

Redundant, or "dead", code imposes a burden on those reading or maintaining the software project. It can make it harder to understand the structure of the code, as well as increasing the complexity of adding new features or fixing bugs. It can also affect compilation and build times for the project, as dead code will still be compiled and built even if it is never used. In some cases it may also affect runtime performance - for example, fields that are written to but never read from, where the value written to the field is expensive to compute. Removing dead code should not change the meaning of the program.

Classes are considered dead if at runtime:

* No methods declared in the class, or a sub-type, are called.
* No fields declared in the class, or a sub-type, are read.
* The class is never constructed.
Any class which is not dead is considered to be "live". Nested classes are considered and listed separately, as a live nested class within a dead outer class can be moved to a separate file, allowing the outer class to be deleted.

A special exception is made for "namespace classes". A namespace class is used only to group static fields, methods and nested classes - it is never instantiated, has no public constructor and has no instance methods. If a class is considered to be a namespace class, then it is live if at least one of the static members of that class is live - including static nested classes.

A class, method, or field may be dead even if it has dependencies from other parts of the program, if those dependencies are from code that is also considered to be dead. We can also consider this from the opposite side - an element is live, if and only if there is an entry point - such as a `main` method - that eventually calls the method, reads the field or constructs the class.

When identifying dead code, we make an assumption that the snapshot of the project includes all possible callers of the code. If the project is a library project, this may not be the case, and code may be flagged as dead when it is only used by other projects not included in the snapshot.

You can customize the results by defining additional "entry points" or by identifying fields that are accessed using reflection. You may also wish to "whitelist" classes, methods or fields that should be excluded from the results. Please refer to the Semmle documentation for more information.


## Recommendation
Before making any changes, confirm that the class is not required by verifying that the only dependencies on the class are from other dead classes and methods. This confirmation is necessary because there may be project-specific frameworks or techniques which can introduce hidden dependencies. If this project is for a library, then consider whether the class is part of the external API, and may be used in external projects that are not included in the snapshot.

After confirming that the class is not required, remove the class. You will also need to remove any references to this class, which may, in turn, require removing other unused classes, methods and fields (see Example 1).

Nested classes within this type should be moved, either to a new top-level type, or to another type, unless they are also marked as dead, in which case they can also be removed. Alternatively, if there are some live nested classes within the dead class, the class can be retained by converting all live nested classes to static members, and removing all instance methods and fields, and all dead static members (see Example 2).

If you observe a large number of false positives, you may need to add extra entry points to identify hidden dependencies caused by the use of a particular framework or technique, or to identify library project entry points. Please refer to the Semmle documentation for more information on how to do this.


## Example 1
In the following example, we have a number of classes, and an "entry point" in the form of a main method.


```java
public static void main(String[] args) {
	String firstName = /*...*/; String lastName = /*...*/;
	// Construct a customer
	Customer customer = new Customer();
	// Set important properties (but not the address)
	customer.setName(firstName, lastName);
	// Save the customer
	customer.save();
}

public class Customer {
	private Address address;
	// ...

	// This setter and getter are unused, and so may be deleted.
	public void addAddress(String line1, String line2, String line3) {
		address = new Address(line1, line2, line3);
	}
	public Address getAddress() { return address; }
}

/*
 * This class is only constructed from dead code, and may be deleted.
 */
public class Address {
	// ...
	public Address(String line1, String line2, String line3) {
		// ...
	}
}

```
The class `Customer` is constructed in the main method, and is therefore live. The class `Address` is constructed in `setAddress`, so we might think that it would also be live. However, `setAddress` is never called by the main method, so, assuming that this is the entire program, an `Address` is never constructed at runtime. Therefore, the `Address` class is dead and can be removed without changing the meaning of this program. To delete the `Address` class we will also need to delete the `setAddress` and `getAddress` methods, and the `address` field, otherwise the program will not compile.


## Example 2
In the next example, we have a `CustomerActions` class containing `Action`s that affect customers. For example, this could be a Java Swing application, and the `Action`s could be actions that are available in the user interface.


```java
/*
 * This class is dead because it is never constructed, and the instance methods are not
 * called.
 */
public class CustomerActions {
	public CustomerActions() {
	}

	// This method is never called,
	public Action createAddCustomerAction () {
		return new AddCustomerAction();
	}

	// These two are used directly
	public static class AddCustomerAction extends Action { /* ... */ }
	public static class RemoveCustomerAction extends Action { /* ... */ }
}

public static void main(String[] args) {
	// Construct the actions directly
	Action action = new CustomerActions.AddCustomerAction();
	action.run();
	Action action = new CustomerActions.RemoveCustomerAction();
	action.run();
}

```
The `CustomerActions` class has a constructor and an instance method, which are never called. Instead, actions are instantiated directly. Although this makes the nested `Action` classes live, live nested classes do not make the outer class live. Therefore, the `CustomerActions` class is marked as dead.

There are two ways to resolve the dead `CustomerActions` class:

* Move each nested static action that is used by the program to a new file, or nest it within a different class, then delete the dead `CustomerActions` class.
* Convert the `CustomerActions` class to a *namespace class*. First convert the constructor to a *suppressed constructor* by making it private, preventing the class from being instantiated, then remove the instance method `createAddCustomerAction`.
Taking the second approach, this is the final result.


```java
// This class is now live - it is used as a namespace class
public class CustomerActions {
	/*
	 * This constructor is suppressing construction of this class, so is not considered
	 * dead.
	 */
	private CustomerActions() { }
	// These two are used directly
	public static class AddCustomerAction extends Action { /* ... */ }
	public static class RemoveCustomerAction extends Action { /* ... */ }
}

public static void main(String[] args) {
	// Construct the actions directly
	Action action = new CustomerActions.AddCustomerAction();
	action.run();
	Action action = new CustomerActions.RemoveCustomerAction();
	action.run();
}

```

## References
* Wikipedia: [Redundant code](https://en.wikipedia.org/wiki/Redundant_code).
* CERT Java Coding Standard: [MSC56-J. Detect and remove superfluous code and values](https://www.securecoding.cert.org/confluence/display/java/MSC56-J.+Detect+and+remove+superfluous+code+and+values).
* Common Weakness Enumeration: [CWE-561](https://cwe.mitre.org/data/definitions/561.html).
