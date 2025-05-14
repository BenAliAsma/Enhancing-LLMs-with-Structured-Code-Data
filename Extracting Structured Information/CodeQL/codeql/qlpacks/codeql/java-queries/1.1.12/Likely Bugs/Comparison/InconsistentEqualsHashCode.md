# Inconsistent equals and hashCode
A class that overrides only one of `equals` and `hashCode` is likely to violate the contract of the `hashCode` method. The contract requires that `hashCode` gives the same integer result for any two equal objects. Not enforcing this property may cause unexpected results when storing and retrieving objects of such a class in a hashing data structure.


## Recommendation
Usually, both methods should be overridden to ensure that they are consistent.


## Example
In the following example, the class `InconsistentEqualsHashCode` overrides `hashCode` but not `equals`.


```java
public class InconsistentEqualsHashCode {
	private int i = 0;
	public InconsistentEqualsHashCode(int i) {
		this.i = i;
	}

	public int hashCode() {
		return i;
	}
}
```
In the following example, the class `InconsistentEqualsHashCodeFix` overrides both `hashCode` and `equals`.


```java
public class InconsistentEqualsHashCodeFix {
	private int i = 0;
	public InconsistentEqualsHashCodeFix(int i) {
		this.i = i;
	}

	@Override
	public int hashCode() {
		return i;
	}

	@Override
	public boolean equals(Object obj) {
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		InconsistentEqualsHashCodeFix that = (InconsistentEqualsHashCodeFix) obj;
		return this.i == that.i;
	}
}
```

## References
* J. Bloch, *Effective Java (second edition)*, Item 9. Addison-Wesley, 2008.
* Java API Specification: [Object.equals](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/Object.html#equals(java.lang.Object)), [Object.hashCode](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/Object.html#hashCode()).
* IBM developerWorks: [Java theory and practice: Hashing it out](https://www.ibm.com/developerworks/java/library/j-jtp05273/index.html).
* Help - Eclipse Platform: [Java Compiler Errors/Warnings Preferences](https://help.eclipse.org/2020-12/advanced/content.jsp?topic=/org.eclipse.jdt.doc.user/reference/preferences/java/compiler/ref-preferences-errors-warnings.htm).
* Common Weakness Enumeration: [CWE-581](https://cwe.mitre.org/data/definitions/581.html).
