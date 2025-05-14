# Feature envy
*Feature envy* refers to situations where a method is "in the wrong place", because it does not use many methods or variables of its own class, but uses a whole range of methods or variables from some other class. This violates the principle of putting data and behavior in the same place, and exposes internals of the other class to the method.


## Recommendation
For each method that may exhibit feature envy, see if it needs to be declared in its present location, or if you can move it to the class it is "envious" of. A common example is a method that calls a large number of getters on another class to perform a calculation that does not rely on anything from its own class. In such cases, you should move the method to the class containing the data. If the calculation depends on some values from the method's current class, they can either be passed as arguments or accessed using getters from the other class.

If it is inappropriate to move the entire method, see if all the dependencies on the other class are concentrated in just one part of the method. If so, you can move them into a method of their own. You can then move this method to the other class and call it from the original method.

If a class is envious of functionality defined in a superclass, perhaps the superclass needs to be rewritten to become more extensible and allow its subtypes to define new behavior without them depending so deeply on the superclass's implementation. The *template method* pattern may be useful in achieving this.

Modern IDEs provide several refactorings that may be useful in addressing instances of feature envy, typically under the names of "Move method" and "Extract method".

Occasionally, behavior can be misinterpreted as feature envy when in fact it is justified. The most common examples are complex design patterns like *visitor* or *strategy*, where the goal is to separate data from behavior.


## Example
In the following example, initially the method `getTotalPrice` is in the `Basket` class, but it only uses data belonging to the `Item` class. Therefore, it represents an instance of feature envy. To refactor it, `getTotalPrice` can be moved to `Item` and its parameter can be removed. The resulting code is easier to understand and keep consistent.


```java
// Before refactoring:
class Item { .. }
class Basket {
	// ..
	float getTotalPrice(Item i) {
		float price = i.getPrice() + i.getTax();
		if (i.isOnSale())
			price = price - i.getSaleDiscount() * price;
		return price;
	}
}

// After refactoring:
class Item {
	// ..
	float getTotalPrice() {
		float price = getPrice() + getTax();
		if (isOnSale())
			price = price - getSaleDiscount() * price;
		return price;
	}
}
```
The refactored code is still appropriate, even if some data from the `Basket` class is necessary for the computation of the total price. For example, if the `Basket` class applies a bulk discount when a sufficient number of items are in the basket, an "additional discount" parameter can be added to `Item.getTotalPrice(..)`. Alternatively, the application of the discount can be performed in a method in `Basket` that calls `Item.getTotalPrice`.


## References
* E. Gamma, R. Helm, R. Johnson, J. Vlissides, *Design patterns: elements of reusable object-oriented software*. Addison-Wesley Longman Publishing Co., Inc., Boston, MA, 1995.
* W. C. Wake, *Refactoring Workbook*, pp. 93&ndash;94. Addison-Wesley Professional, 2004.
