# Inefficient use of key set iterator
Java's Collections Framework provides several different ways of iterating the contents of a map. You can retrieve the set of keys, the collection of values, or the set of "entries" (which are, in effect, key/value pairs).

The choice of iterator can affect performance. For example, it is considered bad practice to iterate the key set of a map if the body of the loop performs a map lookup on each retrieved key anyway.


## Recommendation
Evaluate the requirements of the loop body. If it does not actually need the key apart from looking it up in the map, iterate the map's values (obtained by a call to `values`) instead. If the loop genuinely needs both key and value for each mapping in the map, iterate the entry set (obtained by a call to `entrySet`) and retrieve the key and value from each entry. This saves a more expensive map lookup each time.


## Example
In the following example, the first version of the method `findId` iterates the map `people` using the key set. This is inefficient because the body of the loop needs to access the value for each key. In contrast, the second version iterates the map using the entry set because the loop body needs both the key and the value for each mapping.


```java
// AVOID: Iterate the map using the key set.
class AddressBook {
	private Map<String, Person> people = ...;
	public String findId(String first, String last) {
		for (String id : people.keySet()) {
			Person p = people.get(id);
			if (first.equals(p.firstName()) && last.equals(p.lastName()))
				return id;
		}
		return null;
	}
}

// GOOD: Iterate the map using the entry set.
class AddressBook {
	private Map<String, Person> people = ...;
	public String findId(String first, String last) {
		for (Entry<String, Person> entry: people.entrySet()) {
			Person p = entry.getValue();
			if (first.equals(p.firstName()) && last.equals(p.lastName()))
				return entry.getKey();
		}
		return null;
	}
}
```

## References
* Java API Specification: [Map.entrySet()](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/util/Map.html#entrySet()).
