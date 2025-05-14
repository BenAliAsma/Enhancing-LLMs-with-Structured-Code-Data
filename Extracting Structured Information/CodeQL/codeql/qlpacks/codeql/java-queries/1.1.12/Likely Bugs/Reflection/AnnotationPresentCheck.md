# AnnotationPresent check
To be able to use the `isAnnotationPresent` method on an `AnnotatedElement` at runtime, an annotation must be explicitly annotated with a `RUNTIME` retention policy. Otherwise, the annotation is not retained at runtime and cannot be observed using reflection.


## Recommendation
Explicitly annotate annotations with a `RUNTIME` retention policy if you want to observe their presence using `AnnotatedElement.isAnnotationPresent` at runtime.


## Example
In the following example, the call to `isAnnotationPresent` returns `false` because the annotation cannot be observed using reflection.


```java
public class AnnotationPresentCheck {
	public static @interface UntrustedData { }

	@UntrustedData
	public static String getUserData() {
		Scanner scanner = new Scanner(System.in);
		return scanner.nextLine();
	}

	public static void main(String[] args) throws NoSuchMethodException, SecurityException {
		String data = getUserData();
		Method m = AnnotationPresentCheck.class.getMethod("getUserData");
		if(m.isAnnotationPresent(UntrustedData.class)) {  // Returns 'false'
			System.out.println("Not trusting data from user.");
		}
	}
}
```
To correct this, the annotation is annotated with a `RUNTIME` retention policy.


```java
public class AnnotationPresentCheckFix {
	@Retention(RetentionPolicy.RUNTIME)  // Annotate the annotation
	public static @interface UntrustedData { }

	@UntrustedData
	public static String getUserData() {
		Scanner scanner = new Scanner(System.in);
		return scanner.nextLine();
	}

	public static void main(String[] args) throws NoSuchMethodException, SecurityException {
		String data = getUserData();
		Method m = AnnotationPresentCheckFix.class.getMethod("getUserData");
		if(m.isAnnotationPresent(UntrustedData.class)) {  // Returns 'true'
			System.out.println("Not trusting data from user.");
		}
	}
}
```

## References
* Java API Specification: [Annotation Type Retention](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/annotation/Retention.html), [RetentionPolicy.RUNTIME](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/annotation/RetentionPolicy.html#RUNTIME), [AnnotatedElement.isAnnotationPresent()](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/reflect/AnnotatedElement.html#isAnnotationPresent(java.lang.Class)).
