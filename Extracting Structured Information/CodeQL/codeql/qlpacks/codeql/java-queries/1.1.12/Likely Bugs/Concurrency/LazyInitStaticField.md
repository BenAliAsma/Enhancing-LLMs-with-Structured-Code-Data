# Incorrect lazy initialization of a static field
The tactic of initializing a static field the first time it is used, known as "lazy initialization", can be problematic in a multi-threaded context when used without proper synchronization. If a separate thread starts executing before the field is initialized, the thread may see an incompletely initialized object.


## Recommendation
If lazy initialization is desirable for performance reasons, the best solution is usually to declare the enclosing method `synchronized`. Otherwise, avoid lazy initialization and initialize static fields using static initializers. A third possibility is to declare the field `volatile` and use the double-checked locking idiom as explained in the article referenced below. As the article points out, it is crucial to declare the field `volatile`: double-checked locking by itself is *not* correct under the Java memory model.


## Example
In the following example, the static field `resource` is initialized without synchronization.


```java
class Singleton {
    private static Resource resource;

    public Resource getResource() {
        if(resource == null)
            resource = new Resource();  // Lazily initialize "resource"
        return resource;
    }
}
```
In the following modification of the above example, `Singleton` uses the recommended approach of using a static initializer to initialize `resource`.


```java
class Singleton {
    private static Resource resource;

    static {
        resource = new Resource();  // Initialize "resource" only once
    }
 
    public Resource getResource() {
        return resource;
    }
}
```

## References
* University of Maryland Department of Computer Science: [The "Double-Checked Locking is Broken" Declaration](http://www.cs.umd.edu/~pugh/java/memoryModel/DoubleCheckedLocking.html).
* Common Weakness Enumeration: [CWE-543](https://cwe.mitre.org/data/definitions/543.html).
* Common Weakness Enumeration: [CWE-609](https://cwe.mitre.org/data/definitions/609.html).
