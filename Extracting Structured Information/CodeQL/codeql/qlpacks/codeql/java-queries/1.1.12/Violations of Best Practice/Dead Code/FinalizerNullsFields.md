# Finalizer nulls fields
A finalizer does not need to set an object's fields to `null` to help the garbage collector. At the point in the Java object life-cycle when the `finalize` method is called, the object is no longer reachable from the garbage collection roots. Explicitly setting the object's fields to `null` does not cause the referenced objects to be collected by the garbage collector any earlier, and may even adversely affect performance.

The life-cycle of a Java object has 7 stages:

* **Created** : Memory is allocated for the object and the initializers and constructors have been run.
* **In use** : The object is reachable through a chain of strong references from a garbage collection root. A garbage collection root is a special class of variable (which includes variables on the stack of any thread, static variables of any class, and references from Java Native Interface code).
* **Invisible** : The object has already gone out of scope, but the stack frame of the method that contained the scope is still in memory. Not all objects transition into this state.
* **Unreachable** : The object is no longer reachable through a chain of strong references. It becomes a candidate for garbage collection.
* **Collected** : The garbage collector has identified that the object can be deallocated. If it has a finalizer, it is marked for finalization. Otherwise, it is deallocated.
* **Finalized** : An object with a `finalize` method transitions to this state after the finalize method is completed and the object still remains unreachable.
* **Deallocated** : The object is a candidate for deallocation.
The call to the `finalize` method occurs when the object is in the 'Collected' stage. At that point, it is already unreachable from the garbage collection roots so any of its references to other objects no longer contribute to their reference counts.


## Recommendation
Ensure that the finalizer does not contain any `null` assignments because they are unlikely to help garbage collection.

If a finalizer does nothing but nullify an object's fields, it is best to completely remove the finalizer. Objects with finalizers severely affect performance, and you should avoid defining `finalize` where possible.


## Example
In the following example, `finalize` unnecessarily assigns the object's fields to null.


```java
class FinalizedClass {
	Object o = new Object();
	String s = "abcdefg";
	Integer i = Integer.valueOf(2);
	
	@Override
	protected void finalize() throws Throwable {
		super.finalize();
		//No need to nullify fields
		this.o = null;
		this.s = null;
		this.i = null;
	}
}
```

## References
* J. Bloch, *Effective Java (second edition)*, Item 7. Addison-Wesley, 2008.
* IBM developerWorks: [Java theory and practice: Explicit nulling](https://web.archive.org/web/20201111184342/https://www.ibm.com/developerworks/java/library/j-jtp01274/index.html#3.2).
* Oracle Technology Network: [ How to Handle Java Finalization's Memory-Retention Issues ](https://www.oracle.com/technical-resources/articles/javase/finalization.html).
* S. Wilson and J. Kesselman, *Java Platform Performance: Strategies and Tactics, 1st ed.*, Appendix A. Prentice Hall, 2001.
