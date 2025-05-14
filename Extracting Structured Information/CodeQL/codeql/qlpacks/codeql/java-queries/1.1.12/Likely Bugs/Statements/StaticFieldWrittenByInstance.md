# Static field written by instance method
A static field represents state shared between all instances of a particular class. Typically, static methods are provided to manipulate this static state, and it is bad practice to modify the static state of a class from an instance method (or from a constructor).

There are several reasons why this is bad practice. It can be very difficult to keep the static state consistent when there are multiple instances through which it could be modified. Such modifications represent a readability issue: most programmers would expect a static method to affect static state, and an instance method to affect instance state.


## Recommendation
If the field should be an instance field, ensure that it does not have a `static` modifier.

If the field does have to be static, evaluate the assumptions in the code. Does the field really have to be modified directly in an instance method? It might be better to access the field from within static methods, so that any concerns about synchronization can be addressed without numerous synchronization statements in the code. Perhaps the field modification is part of the static initialization of the class, and should be moved to a static initializer or method.


## Example
In the following example, a static field, `customers`, is written to by an instance method, `initialize`. It is entirely reasonable for another developer to assume that an instance method called `initialize` should be called on each new instance, and that is what the code in `Department` does. Unfortunately, the static field is shared between all instances of `Customer`, and so each time `initialize` is called, the old state is lost.


```java
public class Customer {
	private static List<Customer> customers;
	public void initialize() {
		// AVOID: Static field is written to by instance method.
		customers = new ArrayList<Customer>();
		register();
	}
	public static void add(Customer c) {
		customers.add(c);
	}
}

// ...
public class Department {
	public void addCustomer(String name) {
		Customer c = new Customer(n);
		// The following call overwrites the list of customers
		// stored in 'Customer' (see above).
		c.initialize();
		Customer.add(c);
	}
}

```
The solution is to extract the static initialization of `customers` to a static method, where it will happen exactly once.


## References
* Java Language Specification: [8.3.1.1 static Fields](https://docs.oracle.com/javase/specs/jls/se11/html/jls-8.html#jls-8.3.1.1).
