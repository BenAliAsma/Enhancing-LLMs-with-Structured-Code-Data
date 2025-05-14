# Futile synchronization on field
A block of code that synchronizes on a field and updates that field while the lock is held is unlikely to provide the desired thread safety. Such a synchronized block does not prevent multiple unsynchronized assignments to that field because it obtains a lock on the object stored *in* the field rather than the field itself.


## Recommendation
Instead of synchronizing on the field itself, consider synchronizing on a separate lock object when you want to avoid simultaneous updates to the field. You can do this by declaring a synchronized method and using it for any field updates.


## Example
In the following example, in class A, synchronization takes place on the field that is updated in the body of the `setField` method.


```java
public class A {
    private Object field;  
    
    public void setField(Object o){
        synchronized (field){    // BAD: synchronize on the field to be updated
            field = o;
            // ... more code ...          
        }
    }
}
```
In class B, the recommended approach is shown, where synchronization takes place on a separate lock object.


```java
public class B {
   private final Object lock = new Object();
   private Object field;

   public void setField(Object o){
       synchronized (lock){      // GOOD: synchronize on a separate lock object
           field = o;
           // ... more code ...
       }
   }
}
```

## References
* Java Language Specification: [The synchronized Statement](https://docs.oracle.com/javase/specs/jls/se11/html/jls-14.html#jls-14.19), [synchronized Methods](https://docs.oracle.com/javase/specs/jls/se11/html/jls-8.html#jls-8.4.3.6).
* The Java Tutorials: [Lock Objects](https://docs.oracle.com/javase/tutorial/essential/concurrency/newlocks.html).
* Common Weakness Enumeration: [CWE-662](https://cwe.mitre.org/data/definitions/662.html).
