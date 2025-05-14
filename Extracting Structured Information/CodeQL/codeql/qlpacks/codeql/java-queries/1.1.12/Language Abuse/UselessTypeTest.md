# Useless type test
It is always the case that, for any type `B`, an instance of a type derived from `B` is also an instance of `B`. There is no need to explicitly test that this relationship exists.


## Recommendation
Remove the unnecessary type test to simplify the code.


## Example
The following example shows an unnecessary type test.


```java
public class UselessTypeTest {
	private static class B {}
	private static class D extends B {}

	public static void main(String[] args) {
		D d = new D();
		if(d instanceof B) {	// violation
			System.out.println("Mon dieu, d is a B!");
		}
	}
}
```
