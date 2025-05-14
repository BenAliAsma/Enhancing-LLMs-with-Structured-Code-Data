# Call to Iterator.remove may fail
The `remove` method of the `Iterator` interface is an optional operation. It is not supported by iterators on unmodifiable collections, or iterators on lists constructed by the `Arrays.asList` method. Invoking `remove` on such an iterator will lead to an `UnsupportedOperationException`.


## Recommendation
If a collection is meant to be modified after construction, use a modifiable collection type such as `ArrayList` or `HashSet`.


## Example
In the following example, the constructor `A(Integer...)` initializes the field `A.l` to `Arrays.asList(is)`. While the type of lists returned by `Arrays.asList` supports element updates through the `set` method, it does not support element removal. Hence the call to `iter.remove` on line 20 must fail at runtime.


```java
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

public class A {
	private List<Integer> l;
	
	public A(Integer... is) {
		this.l = Arrays.asList(is);
	}
	
	public List<Integer> getList() {
		return l;
	}

	public static void main(String[] args) {
		A a = new A(23, 42);
		for (Iterator<Integer> iter = a.getList().iterator(); iter.hasNext();)
			if (iter.next()%2 != 0)
				iter.remove();
	}
}

```
To avoid this failure, copy the list returned by `Arrays.asList` into a newly created `ArrayList` like this:


```java
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.ArrayList;

public class A {
	private List<Integer> l;
	
	public A(Integer... is) {
		this.l = new ArrayList<Integer>(Arrays.asList(is));
	}
	
	public List<Integer> getList() {
		return l;
	}

	public static void main(String[] args) {
		A a = new A(23, 42);
		for (Iterator<Integer> iter = a.getList().iterator(); iter.hasNext();)
			if (iter.next()%2 != 0)
				iter.remove();
	}
}

```

## References
* Mark Needham: [Java: Fooled by java.util.Arrays.asList](https://dzone.com/articles/java-fooled).
