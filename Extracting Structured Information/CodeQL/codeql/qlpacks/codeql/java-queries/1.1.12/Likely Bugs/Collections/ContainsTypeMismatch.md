# Type mismatch on container access
The `contains` method of the `Collection` interface has an argument of type `Object`. Therefore, you can try to check if an object of any type is a member of a collection, regardless of the collection's element type. However, although you can call `contains` with an argument of a different type than that of the collection, it is unlikely that the collection actually contains an object of this type.

Similar considerations apply to other container access methods, such as `Map.get`, where the argument may also have type `Object`.


## Recommendation
Ensure that you use the correct argument with a call to `contains`.


## Example
In the following example, although the argument to `contains` is an integer, the code does not result in a type error because the argument does not have to match the type of the elements of `list`. However, the argument is unlikely to be found (and the body of the `if` statement is therefore not executed), so it is probably a typographical error: the argument should be enclosed in quotation marks.


```java
void m(List<String> list) {
	if (list.contains(123)) {  // Call 'contains' with non-string argument (without quotation marks)
		// ...
	}
}
```
Note that you must take particular care when working with collections over boxed types, as illustrated in the following example. The first call to `contains` returns `false` because you cannot compare two boxed numeric primitives of different types, in this case `Short(1)` (in `set`) and `Integer(1)` (the argument). The second call to `contains` returns `true` because you can compare `Short(1)` and `Short(1)`.


```java
HashSet<Short> set = new HashSet<Short>();
short s = 1;
set.add(s);
// Following statement prints 'false', because the argument is a literal int, which is auto-boxed
// to an Integer
System.out.println(set.contains(1));
// Following statement prints 'true', because the argument is a literal int that is cast to a short, 
// which is auto-boxed to a Short
System.out.println(set.contains((short)1));

```

## References
* Java API Specification: [Collection.contains](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/util/Collection.html#contains(java.lang.Object)).
