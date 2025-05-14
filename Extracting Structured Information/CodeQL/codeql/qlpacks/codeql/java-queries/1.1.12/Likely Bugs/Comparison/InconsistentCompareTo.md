# Inconsistent compareTo
A class that overrides `compareTo` but not `equals` may not implement a natural ordering that is consistent with `equals`.


## Recommendation
Although this consistency is not strictly required by the `compareTo` contract, usually both methods should be overridden to ensure that they are consistent, that is, that `x.compareTo(y)==0` is `true` if and only if `x.equals(y)` is `true`, for any non-null `x` and `y`.


## Example
In the following example, the class `InconsistentCompareTo` overrides `compareTo` but not `equals`.


```java
public class InconsistentCompareTo implements Comparable<InconsistentCompareTo> {
	private int i = 0;
	public InconsistentCompareTo(int i) {
		this.i = i;
	}
	
	public int compareTo(InconsistentCompareTo rhs) {
		return i - rhs.i;
	}
}
```
In the following example, the class `InconsistentCompareToFix` overrides both `compareTo` and `equals`.


```java
public class InconsistentCompareToFix implements Comparable<InconsistentCompareToFix> {
	private int i = 0;
	public InconsistentCompareToFix(int i) {
		this.i = i;
	}
	
	public int compareTo(InconsistentCompareToFix rhs) {
		return i - rhs.i;
	}

	public boolean equals(InconsistentCompareToFix rhs) {
		return i == rhs.i;
	}
}
```
If you require a natural ordering that is inconsistent with `equals`, you should document it clearly.


## References
* J. Bloch, *Effective Java (second edition)*, Item 12. Addison-Wesley, 2008.
* Java API Specification: [Comparable.compareTo](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/Comparable.html#compareTo(T)), [Object.equals](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/Object.html#equals(java.lang.Object)).
