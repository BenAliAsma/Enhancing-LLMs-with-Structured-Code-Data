# Incorrect absolute value of random number
Using `Math.abs` on the result of a call to `Random.nextInt()` (or `Random.nextLong()`) is not guaranteed to return a non-negative number. `Random.nextInt()` can return `Integer.MIN_VALUE`, which when passed to `Math.abs` results in the same value, `Integer.MIN_VALUE`. (Because of the two's-complement representation of integers in Java, the positive equivalent of `Integer.MIN_VALUE` cannot be represented in the same number of bits.) The case for `Random.nextLong()` is similar.


## Recommendation
If a non-negative random integer is required, use `Random.nextInt(int)` instead, and use `Integer.MAX_VALUE` as its parameter. The values that might be returned do not include `Integer.MAX_VALUE` itself, but this solution is likely to be sufficient for most purposes.

Another solution is to increment the value of `Random.nextInt()` by one, if it is negative, before passing the result to `Math.abs`. This solution has the advantage that `0` has the same probability as other numbers.


## Example
In the following example, `mayBeNegativeInt` is negative if `nextInt` returns `Integer.MIN_VALUE`. The example shows how using the two solutions described above means that `positiveInt` is always assigned a positive number.


```java
public static void main(String args[]) {
    Random r = new Random();

    // BAD: 'mayBeNegativeInt' is negative if
    // 'nextInt()' returns 'Integer.MIN_VALUE'.
    int mayBeNegativeInt = Math.abs(r.nextInt());

    // GOOD: 'nonNegativeInt' is always a value between 0 (inclusive)
    // and Integer.MAX_VALUE (exclusive).
    int nonNegativeInt = r.nextInt(Integer.MAX_VALUE);

    // GOOD: When 'nextInt' returns a negative number increment the returned value.
    int nextInt = r.nextInt();
    if(nextInt < 0)
        nextInt++;
    int nonNegativeInt = Math.abs(nextInt);
}

```

## References
* Java API Specification: [Math.abs(int)](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/Math.html#abs(int)), [Math.abs(long)](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/Math.html#abs(long)), [Random](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/util/Random.html).
* Java Language Specification: [4.2.1 Integral Types and Values](https://docs.oracle.com/javase/specs/jls/se11/html/jls-4.html#jls-4.2.1).
