# Race condition in double-checked locking object initialization
Double-checked locking is a common pattern for lazy initialization of a field accessed by multiple threads. Depending on the memory model of the underlying runtime, it can, however, be quite difficult to implement correctly, since reorderings performed by compiler, runtime, or CPU might expose un-initialized or half-way initialized objects to other threads. Java has since version 5 improved its memory model to support double-checked locking if the underlying field is marked `volatile` and if all initialization happens before the volatile write.


## Recommendation
First, it should be considered whether the getter that performs the lazy initialization is performance critical. If not, a much simpler solution is to completely avoid double-checked locking and simply mark the entire getter as `synchronized`. This is much easier to get right and guards against hard-to-find concurrency bugs.

If double-checked locking is used, it is important that the underlying field is `volatile` and that the update to the field is the last thing that happens in the synchronized region, that is, all initialization must be done before the field is assigned. Furthermore, the Java version must be 5 or newer. Reading a `volatile` field has a slight overhead, so it is also useful to use a local variable to minimize the number of volatile reads.


## Example
The following code lazily initializes `f` to `new MyObject()`.


```java
private Object lock = new Object();
private MyObject f = null;

public MyObject getMyObject() {
  if (f == null) {
    synchronized(lock) {
      if (f == null) {
        f = new MyObject(); // BAD
      }
    }
  }
  return f;
}

```
This code is not thread-safe as another thread might see the assignment to `f` before the constructor finishes evaluating, for example if the compiler inlines the memory allocation and the constructor and reorders the assignment to `f` to occur just after the memory allocation.

Another example that also is not thread-safe, even when `volatile` is used, is if additional initialization happens after the assignment to `f`, since then other threads may access the constructed object before it is fully initialized, even without any reorderings by the compiler or runtime.


```java
private Object lock = new Object();
private volatile MyObject f = null;

public MyObject getMyObject() {
  if (f == null) {
    synchronized(lock) {
      if (f == null) {
        f = new MyObject();
        f.init(); // BAD
      }
    }
  }
  return f;
}

```
The code above should be rewritten to both use `volatile` and finish all initialization before `f` is updated. Additionally, a local variable can be used to avoid reading the field more times than necessary.


```java
private Object lock = new Object();
private volatile MyObject f = null;

public MyObject getMyObject() {
  MyObject result = f;
  if (result == null) {
    synchronized(lock) {
      result = f;
      if (result == null) {
        result = new MyObject();
        result.init();
        f = result; // GOOD
      }
    }
  }
  return result;
}

```
As a final note, it is possible to use double-checked locking correctly without `volatile` if the object you construct is immutable (that is, the object declares all fields as `final`), and the double-checked field is read exactly once outside the synchronized block.

Given that all fields in `MyImmutableObject` are declared `final` then the following example is protected against exposing uninitialized fields to another thread. However, since there are two reads of `f` without synchronization, it is possible that these are reordered, which means that this method can return `null`.


```java
private Object lock = new Object();
private MyImmutableObject f = null;

public MyImmutableObject getMyImmutableObject() {
  if (f == null) {
    synchronized(lock) {
      if (f == null) {
        f = new MyImmutableObject();
      }
    }
  }
  return f; // BAD
}

```
In this case, using a local variable to minimize the number of field reads is no longer a performance improvement, but rather a crucial detail that is necessary for correctness.


## References
* [The "Double-Checked Locking is Broken" Declaration](https://www.cs.umd.edu/~pugh/java/memoryModel/DoubleCheckedLocking.html).
* Java Language Specification: [17.4. Memory Model](https://docs.oracle.com/javase/specs/jls/se11/html/jls-17.html#jls-17.4).
* Wikipedia: [Double-checked locking](https://en.wikipedia.org/wiki/Double-checked_locking).
* Aleksey Shipilëv: [Safe Publication and Safe Initialization in Java](https://shipilev.net/blog/2014/safe-public-construction/).
* Aleksey Shipilëv: [Close Encounters of The Java Memory Model Kind](https://shipilev.net/blog/2016/close-encounters-of-jmm-kind/).
* Common Weakness Enumeration: [CWE-609](https://cwe.mitre.org/data/definitions/609.html).
