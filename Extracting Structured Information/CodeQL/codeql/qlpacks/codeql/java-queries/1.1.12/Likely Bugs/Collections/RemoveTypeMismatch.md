# Type mismatch on container modification
The `remove` method of the `Collection` interface has an argument of type `Object`. Therefore, you can try to remove an object of any type from a collection, regardless of the collection's element type. However, although you can call `remove` with an argument of a different type than that of the collection, it is unlikely that the collection actually contains an object of this type.

Similar considerations apply to other container modification methods, such as `Map.remove`, where the argument may also have type `Object`.


## Recommendation
Ensure that you use the correct argument with a call to `remove`.


## Example
In the following example, although the argument to `contains` is an integer, the code does not result in a type error because the argument to `remove` does not have to match the type of the elements of `list`. However, the argument is unlikely to be found and removed (and the body of the `if` statement is therefore not executed), so it is probably a typographical error: the argument should be enclosed in quotation marks.


```java
void m(List<String> list) {
	if (list.remove(123)) {  // Call 'remove' with non-string argument (without quotation marks)
		// ...
	}
}
```
Note that you must take particular care when working with collections over boxed types, as illustrated in the following example. The first call to `remove` fails because you cannot compare two boxed numeric primitives of different types, in this case `Short(1)` (in `set`) and `Integer(1)` (the argument). Therefore, `remove` cannot find the item to remove. The second call to `remove` succeeds because you can compare `Short(1)` and `Short(1)`. Therefore, `remove` can find the item to remove.


```java
HashSet<Short> set = new HashSet<Short>();
short s = 1;
set.add(s);
// Following statement fails, because the argument is a literal int, which is auto-boxed 
// to an Integer
set.remove(1);
System.out.println(set); // Prints [1]
// Following statement succeeds, because the argument is a literal int that is cast to a short, 
// which is auto-boxed to a Short
set.remove((short)1);
System.out.println(set); // Prints []
```

## References
* Java API Specification: [Collection.remove](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/util/Collection.html#remove(java.lang.Object)).
