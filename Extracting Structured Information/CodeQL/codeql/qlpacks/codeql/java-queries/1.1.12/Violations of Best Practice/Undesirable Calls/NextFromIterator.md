# Next in hasNext implementation
Iterator implementations with a `hasNext` method that calls the `next` method are most likely incorrect. This is because `next` changes the iterator's position to the next element and returns that element, which is unlikely to be desirable in the implementation of `hasNext`.


## Recommendation
Ensure that any calls to `next` from within `hasNext` are legitimate. The `hasNext` method should indicate whether there are further elements remaining in the iteration without changing the iterator's state by calling `next`.


## Example
In the following example, which outputs the contents of a string, `hasNext` calls `next`, which has the effect of changing the iterator's position. Given that `main` also calls `next` when it outputs an item, some items are skipped and only half the items are output.


```java
public class NextFromIterator implements Iterator<String> {
	private int position = -1;
	private List<String> list = new ArrayList<String>() {{
		add("alpha"); add("bravo"); add("charlie"); add("delta"); add("echo"); add("foxtrot");
	}};
	
	public boolean hasNext() {
		return next() != null;  // BAD: Call to 'next'
	}
	
	public String next() {
		position++;
		return position < list.size() ? list.get(position) : null;
	}

	public void remove() {
		// ...
	}
	
	public static void main(String[] args) {
		NextFromIterator x = new NextFromIterator();
		while(x.hasNext()) {
			System.out.println(x.next());
		}
	}
}
```
Instead, the implementation of `hasNext` should use another way of indicating whether there are further elements in the string without calling `next`. For example, `hasNext` could check the underlying array directly to see if there is an element at the next position.


## References
* Java API Specification: [Iterator.hasNext()](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/util/Iterator.html#hasNext()), [Iterator.next()](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/util/Iterator.html#next()).
