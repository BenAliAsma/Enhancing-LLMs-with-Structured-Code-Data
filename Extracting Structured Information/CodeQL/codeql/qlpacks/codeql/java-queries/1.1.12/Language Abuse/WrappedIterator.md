# Iterable wrapping an iterator
Java has two interfaces for dealing with iteration, `Iterable<T>` and `Iterator<T>`. An `Iterable<T>` represents a sequence of elements that can be traversed, and an `Iterator<T>` represents the **state** of an ongoing traversal. As an example, all the `Collection<T>` classes in the Java standard library implement `Iterable<T>`. Comparing this to a traditional `for` loop that increments an integer index and iterates over the elements of an array, then the `Iterable<T>` object corresponds to the array, whereas the `Iterator<T>` object corresponds to the index variable.

Implementations of `Iterable<T>` are generally expected to support multiple traversals of the element sequence they represent, although there can be exceptions if the underlying data somehow makes this undesirable, see for example `DirectoryStream<T>`. If an implementation of `Iterable<T>` does not support multiple iterations, then its `iterator()` method should throw an exception on its second and subsequent calls. This makes bugs easier to find if such an `Iterable<T>` is used more than once, for example in two different for-each loops.


## Recommendation
When writing the `iterator()` method in an `Iterable<T>` then it is important to make sure that each call will result in a fresh `Iterator<T>` instance containing all the necessary state for keeping track of the iteration. If the iterator is stored in the `Iterable<T>`, or somehow refers to iteration state stored in the `Iterable<T>`, then subsequent calls to `iterator()` can result in loops that only traverse a subset of the elements or have no effect at all.


## Example
The following example returns the same iterator on every call, and therefore causes the second loop to terminate immediately without any effect.


```java
class MySequence implements Iterable<MyElem> {
  // ... some reference to data
  final Iterator<MyElem> it = data.iterator();
  // Wrong: reused iterator
  public Iterator<MyElem> iterator() {
    return it;
  }
}

void useMySequence(MySequence s) {
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
This second example returns a newly created iterator each time, but still relies on iteration state stored in the surrounding class, and therefore also causes the second loop to terminate immediately.


```java
class MySequence implements Iterable<MyElem> {
  // ... some reference to data
  final Iterator<MyElem> it = data.iterator();
  // Wrong: iteration state outside returned iterator
  public Iterator<MyElem> iterator() {
    return new Iterator<MyElem>() {
      public boolean hasNext() {
        return it.hasNext();
      }
      public MyElem next() {
        return transformElem(it.next());
      }
      public void remove() {
        // ...
      }
    };
  }
}

```
The code should instead be written like this, such that each call to `iterator()` correctly gives a fresh iterator that starts at the beginning.


```java
class MySequence implements Iterable<MyElem> {
  // ... some reference to data
  public Iterator<MyElem> iterator() {
    return new Iterator<MyElem>() {
      // Correct: iteration state inside returned iterator
      final Iterator<MyElem> it = data.iterator();
      public boolean hasNext() {
        return it.hasNext();
      }
      public MyElem next() {
        return transformElem(it.next());
      }
      public void remove() {
        // ...
      }
    };
  }
}

```

## References
* Java Language Specification: [The enhanced for statement](https://docs.oracle.com/javase/specs/jls/se11/html/jls-14.html#jls-14.14.2).
* Java API Specification: [Interface Iterable&lt;T&gt;](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/Iterable.html), [Interface Iterator&lt;T&gt;](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/util/Iterator.html), [Interface DirectoryStream&lt;T&gt;](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/nio/file/DirectoryStream.html).
