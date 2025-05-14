# Random used only once
A program that uses `java.util.Random` to generate a sequence of pseudo-random numbers *should not* create a new instance of `Random` every time a new pseudo-random number is required (for example, `new Random().nextInt()`).

According to the Java API Specification:

> If two instances of `Random` are created with the same seed, and the same sequence of method calls is made for each, they will generate and return identical sequences of numbers.

The sequence of pseudo-random numbers returned by these calls depends only on the value of the seed. If you construct a new `Random` object each time a pseudo-random number is needed, this does not generate a good distribution of pseudo-random numbers, even though the parameterless `Random()` constructor tries to initialize itself with a unique seed.


## Recommendation
Create a `Random` object once and use the same instance when generating sequences of pseudo-random numbers (by calling `nextInt`, `nextLong`, and so on).


## Example
In the following example, generating a series of pseudo-random numbers, such as `notReallyRandom` and `notReallyRandom2`, by creating a new instance of `Random` each time is unlikely to result in a good distribution of pseudo-random numbers. In contrast, generating a series of pseudo-random numbers, such as `random1` and `random2`, by calling `nextInt` each time *is* likely to result in a good distribution. This is because the numbers are based on only one `Random` object.


```java
public static void main(String args[]) {
	// BAD: A new 'Random' object is created every time
	// a pseudo-random integer is required.
	int notReallyRandom = new Random().nextInt();
	int notReallyRandom2 = new Random().nextInt();
	
	// GOOD: The same 'Random' object is used to generate 
	// two pseudo-random integers.
	Random r = new Random();
	int random1 = r.nextInt();
	int random2 = r.nextInt();
}
```

## References
* Java API Specification: [Random](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/util/Random.html).
* Common Weakness Enumeration: [CWE-335](https://cwe.mitre.org/data/definitions/335.html).
