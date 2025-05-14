# Iterator implementing Iterable
Java has two interfaces for dealing with iteration, `Iterable<T>` and `Iterator<T>`. An `Iterable<T>` represents a sequence of elements that can be traversed, and an `Iterator<T>` represents the **state** of an ongoing traversal. As an example, all the `Collection<T>` classes in the Java standard library implement `Iterable<T>`. Comparing this to a traditional `for` loop that increments an integer index and iterates over the elements of an array, then the `Iterable<T>` object corresponds to the array, whereas the `Iterator<T>` object corresponds to the index variable.

Implementations of `Iterable<T>` are generally expected to support multiple traversals of the element sequence they represent, although there can be exceptions if the underlying data somehow makes this undesirable, see for example `DirectoryStream<T>`. If an implementation of `Iterable<T>` does not support multiple iterations, then its `iterator()` method should throw an exception on its second and subsequent calls. This makes bugs easier to find if such an `Iterable<T>` is used more than once, for example in two different for-each loops.


## Recommendation
When working with custom implementations of `Iterator<T>` it is easy to add `implements Iterable<T>` and a simple `return this;` implementation of `iterator()` to support the for-each syntax. This can, however, hide subtle bugs and is therefore not recommended. It is better to separate the two and use a main representation that only implements `Iterable<T>` without containing any iteration state. This object can then return a short-lived `Iterator<T>` each time it needs to be traversed.

If this refactoring is undesirable for some reason, then the `iterator()` method should at the very least throw an exception if called more than once.


## Example
The following example does not distinguish the iterable from its iterator, and therefore causes the second loop to terminate immediately without any effect.


```java
class ElemIterator implements Iterator<MyElem>, Iterable<MyElem> {
  private MyElem[] data;
  private idx = 0;

  public boolean hasNext() {
    return idx < data.length;
  }
  public MyElem next() {
    return data[idx++];
  }
  public Iterator<MyElem> iterator() {
    return this;
  }
  // ...
}

void useMySequence(Iterable<MyElem> s) {
  // do some work by traversing the sequence
  for (MyElem e : s) {
    // ...
  }
  // do some more work by traversing it again
  for (MyElem e : s) {
    // ...
  }
}

```
The best solution is a refactoring along the following lines where `Iterable` classes are used to pass around references to data. This allows the `Iterator` instances to be short-lived and avoids the sharing of iteration state.


```java
class ElemSequence implements Iterable<MyElem> {
  private MyElem[] data;

  public Iterator<MyElem> iterator() {
    return new Iterator<MyElem>() {
      private idx = 0;
      public boolean hasNext() {
        return idx < data.length;
      }
      public MyElem next() {
        return data[idx++];
      }
    };
  }
  // ...
}

```
If a refactoring, as described above, is too cumbersome or is otherwise undesirable, then a guard can be inserted, as shown below. Using a guard ensures that multiple iteration fails early, making it easier to find any related bugs. This solution is less ideal than the refactoring above, but nevertheless an improvement over the original.


```java
class ElemIterator implements Iterator<MyElem>, Iterable<MyElem> {
  private MyElem[] data;
  private idx = 0;
  private boolean usedAsIterable = false;

  public boolean hasNext() {
    return idx < data.length;
  }
  public MyElem next() {
    return data[idx++];
  }
  public Iterator<MyElem> iterator() {
    if (usedAsIterable || idx > 0)
      throw new IllegalStateException();
    usedAsIterable = true;
    return this;
  }
  // ...
}

```

## References
* Java Language Specification: [The enhanced for statement](https://docs.oracle.com/javase/specs/jls/se11/html/jls-14.html#jls-14.14.2).
* Java API Specification: [Interface Iterable&lt;T&gt;](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/Iterable.html), [Interface Iterator&lt;T&gt;](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/util/Iterator.html), [Interface DirectoryStream&lt;T&gt;](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/nio/file/DirectoryStream.html).
