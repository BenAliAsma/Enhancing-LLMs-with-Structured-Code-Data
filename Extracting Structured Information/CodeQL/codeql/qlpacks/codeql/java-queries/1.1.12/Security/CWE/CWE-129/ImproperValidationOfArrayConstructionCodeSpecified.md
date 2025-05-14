# Improper validation of code-specified size used for array construction
Constructing an array using a size that may be zero can result in the creation of an empty array. If an empty array is accessed without further checks, an `ArrayIndexOutOfBoundsException` is thrown.

This can happen when a fixed value of zero, or a random value that may be zero, is used as the size directly.


## Recommendation
The size used in the array initialization should be verified to be greater than zero before being used. Alternatively, the array access may be placed within a conditional that ensures it is only accessed if the index is less than the array size.


## Example
The following program constructs an array with the size specified by some random value:


```java
public class PossibleArrayIndexOutOfBounds {

  public static void main(String[] args) {
      int numberOfItems = new Random().nextInt(10);

      if (numberOfItems >= 0) {
        /*
         * BAD numberOfItems may be zero, which would cause the array indexing operation to
         * throw an ArrayIndexOutOfBoundsException
         */
        String items = new String[numberOfItems];
        items[0] = "Item 1";
      }

      if (numberOfItems > 0) {
        /*
         * GOOD numberOfItems must be greater than zero, so the indexing succeeds.
         */
        String items = new String[numberOfItems];
        items[0] = "Item 1";
      }
  }
}
```
The first array construction is protected by a condition that checks if the random value is zero or more. However, if the random value is `0` then an empty array is created, and any array access would fail with an `ArrayIndexOutOfBoundsException`.

The second array construction is protected by a condition that checks if the random value is greater than zero. The array will therefore never be empty, and the following array access will not throw an `ArrayIndexOutOfBoundsException`.


## References
* Java API Specification: [ArrayIndexOutOfBoundsException](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/ArrayIndexOutOfBoundsException.html).
* Common Weakness Enumeration: [CWE-129](https://cwe.mitre.org/data/definitions/129.html).
