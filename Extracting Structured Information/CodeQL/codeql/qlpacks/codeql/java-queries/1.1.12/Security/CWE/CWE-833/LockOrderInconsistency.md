# Lock order inconsistency
Acquiring two locks in an inconsistent order may result in deadlock if two threads simultaneously attempt to acquire the locks, and each thread succeeds in acquiring one lock prior to being able to acquire the other lock.


## Recommendation
To guard against deadlock, use one of the following alternatives:

* Define an ordering on locks, and ensure that clients respect that ordering when acquiring locks.
* Simplify the code to only use a single lock, where possible.
* Use `ReentrantLock`s and acquire locks using `tryLock()` instead of `lock()`.

## Example
In the following example, one method acquires `primaryLock` followed by `savingsLock`, and another method acquires these locks in reverse order. This may result in deadlock if the two methods are invoked by two threads simultaneously, and each thread acquires one of the two locks prior to being able to acquire the other one.


```java
class Test {
	private int primaryAccountBalance;
	private Object primaryLock = new Object();
	private int savingsAccountBalance;
	private Object savingsLock = new Object();

	public boolean transferToSavings(int amount) {
		synchronized(primaryLock) {
			synchronized(savingsLock) {
				if (amount>0 && primaryAccountBalance>=amount) {
					primaryAccountBalance -= amount;
					savingsAccountBalance += amount;
					return true;
				}
			}
		}
		return false;
	}
	public boolean transferToPrimary(int amount) {
		// AVOID: lock order is different from "transferToSavings"
		// and may result in deadlock
		synchronized(savingsLock) {
			synchronized(primaryLock) {
				if (amount>0 && savingsAccountBalance>=amount) {
					savingsAccountBalance -= amount;
					primaryAccountBalance += amount;
					return true;
				}
			}
		}
		return false;
	}
}

```
One way to address the issue in the above example is to reverse the lock order in `transferToPrimary` to match the lock order in `transferToSecondary`.


## References
* SEI CERT Oracle Coding Standard for Java: [LCK07-J. Avoid deadlock by requesting and releasing locks in the same order](https://wiki.sei.cmu.edu/confluence/display/java/LCK07-J.+Avoid+deadlock+by+requesting+and+releasing+locks+in+the+same+order).
* Java Language Specification: [Synchronization](https://docs.oracle.com/javase/specs/jls/se11/html/jls-17.html#jls-17.1).
* Java API Specification: [ReentrantLock](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/util/concurrent/locks/ReentrantLock.html).
* Common Weakness Enumeration: [CWE-833](https://cwe.mitre.org/data/definitions/833.html).
