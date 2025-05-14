# Synchronization on boxed types or strings
Code should not synchronize on a variable or field of a boxed type (for example `Integer`, `Boolean`) or of type `String` since it is likely to contain an object that is used throughout the program. For example, `Boolean.TRUE` holds a single instance that will be used in many places throughout the program: whenever `true` is autoboxed or a call to `Boolean.valueOf` is made with `true` as an argument the same instance of `Boolean` is returned. It is therefore likely that two classes synchronizing on a field of type `Boolean` will end up synchronizing on the same object. This may lead to deadlock or threads being blocked unnecessarily.


## Recommendation
Synchronize on a specific lock object instead of using an object with a boxed type.


## Example
In the following example, the intention is to allow `ThreadA` and `ThreadB` to run at the same time. Unfortunately, `ThreadA.lock` and `ThreadB.lock` both refer to the same object (that is, the interned value of the `String` `"lock"`) so the synchronized blocks in their run methods can not be executed concurrently.


```java
class BadSynchronize{
		
	class ThreadA extends Thread{
		private String value = "lock"
		
		public void run(){
			synchronized(value){
				//...
			}
		}
	}
	
	class ThreadB extends Thread{
		private String value = "lock"
		
		public void run(){
			synchronized(value){
				//...
			}
		}
	}
	
	public void run(){
		new ThreadA().start();
		new ThreadB().start();
	}
		
}
```
In the following example, the approach recommended above is shown. A separate lock object is created for each thread allowing them to execute concurrently.


```java
class GoodSynchronize{
		
	class ThreadA extends Thread{
		private Object lock = new Object();
		
		public void run(){
			synchronized(lock){
				//...
			}
		}
	}
	
	class ThreadB extends Thread{
		private Object lock = new Object();
				
		public void run(){
			synchronized(lock){
				//...
			}
		}
	}
	
	public void run(){
		new ThreadA().start();
		new ThreadB().start();
	}
		
}
```

## References
* SEI CERT Oracle Coding Standard for Java: [LCK01-J. Do not synchronize on objects that may be reused](https://wiki.sei.cmu.edu/confluence/display/java/LCK01-J.+Do+not+synchronize+on+objects+that+may+be+reused),
* Common Weakness Enumeration: [CWE-662](https://cwe.mitre.org/data/definitions/662.html).
