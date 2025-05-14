# Exposing internal representation
A subtle type of defect is caused when an object accidentally exposes its internal representation to the code outside the object, and the internal representation is then (deliberately or accidentally) modified in ways that the object is not prepared to handle. Most commonly, this happens when a getter returns a direct reference to a mutable field within the object, or a setter just assigns a mutable argument to its field.


## Recommendation
There are three ways of addressing this problem:

* **Using immutable objects** : The fields store objects that are *immutable*, which means that once constructed their value can never be changed. Examples from the standard library are `String`, `Integer` or `Float`. Although such an object may be aliased, or shared between several contexts, there can be no unexpected changes to the internal state of the object because it cannot be modified.
* **Creating a read-only view** : The `java.util.Collections.unmodifiable*` methods can be used to create a read-only view of a collection without copying it. This tends to give better performance than creating copies of objects. Note that this technique is not suitable for every situation, because any changes to the underlying collection will spread to affect the view. This can lead to unexpected results, and is a particular danger when writing multi-threaded code.
* **Making defensive copies** : Each setter (or constructor) makes a copy or clone of the incoming parameter. In this way, it constructs an instance known only internally, and no matter what happens with the object that was passed in, the state stays consistent. Conversely, each getter for a field must also construct a copy of the field's value to return.

## Example
In the following example, the private field `items` is returned directly by the getter `getItems`. Thus, a caller obtains a reference to internal object state and can manipulate the collection of items in the cart. In the example, each of the carts is emptied when `countItems` is called.


```java
public class Cart {
	private Set<Item> items;
	// ...
	// AVOID: Exposes representation
	public Set<Item> getItems() {
		return items;
	}
}
....
int countItems(Set<Cart> carts) {
	int result = 0;
	for (Cart cart : carts) {
		Set<Item> items = cart.getItems();
		result += items.size();
		items.clear(); // AVOID: Changes internal representation
	}
	return result;
}
```
The solution is for `getItems` to return a *copy* of the actual field, for example `return new HashSet<Item>(items);`.


## References
* J. Bloch, *Effective Java (second edition)*, Items 15 and 39. Addison-Wesley, 2008.
* Java API Specification: [Collections](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/util/Collections.html).
* Common Weakness Enumeration: [CWE-485](https://cwe.mitre.org/data/definitions/485.html).
