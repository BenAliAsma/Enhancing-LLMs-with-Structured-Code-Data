# Equals or hashCode on arrays
The `equals` and `hashCode` methods on arrays only consider object identity, not array contents, which is unlikely to be what is intended.


## Recommendation
To compare the lengths of the arrays and the corresponding pairs of elements in the arrays, use one of the comparison methods from `java.util.Arrays`:

* The method `Arrays.equals` performs a shallow comparison. That is, array elements are compared using `equals`.
* The method `Arrays.deepEquals` performs a deep comparison, which is appropriate for comparisons of nested arrays.
Similarly, `Arrays.hashCode` and `Arrays.deepHashCode` can be used to compute shallow and deep hash codes based on the hash codes of individual array elements.


## Example
In the following example, the two arrays are first compared using the `Object.equals` method. Because this checks only reference equality and the two arrays are different objects, `Object.equals` returns `false`. The two arrays are then compared using the `Arrays.equals` method. Because this compares the length and contents of the arrays, `Arrays.equals` returns `true`.


```java
public void arrayExample(){
    String[] array1 = new String[]{"a", "b", "c"};
    String[] array2 = new String[]{"a", "b", "c"};

    // Reference equality tested: prints 'false'
    System.out.println(array1.equals(array2));
    
    // Equality of array elements tested: prints 'true'
    System.out.println(Arrays.equals(array1, array2));
}
```

## References
* Java API Specification: [Arrays.equals](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/util/Arrays.html#equals(java.lang.Object[],java.lang.Object[])), [Arrays.deepEquals](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/util/Arrays.html#deepEquals(java.lang.Object[],java.lang.Object[])), [Objects.deepEquals](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/util/Objects.html#deepEquals(java.lang.Object,java.lang.Object)), [Object.equals](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/Object.html#equals(java.lang.Object)), [Arrays.hashCode](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/util/Arrays.html#hashCode(java.lang.Object[])), [Arrays.deepHashCode](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/util/Arrays.html#deepHashCode(java.lang.Object[])), [Object.hashCode](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/Object.html#hashCode()).
