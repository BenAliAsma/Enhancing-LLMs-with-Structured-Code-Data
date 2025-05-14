# Inconsistent lock sequence
Locks held concurrently should be locked in a consistent sequence, otherwise the program can deadlock. This rule detects nested `lock` statements that lock variables in a different sequence in different parts of the program.


## Recommendation
This problem can be avoided by ensuring that nested `lock` statements always lock variables in the same sequence.


## Example
The following example shows a program running two threads, which deadlocks because `thread1` holds `lock1` and is waiting to acquire `lock2`, whilst `thread2` holds `lock2` and is waiting to acquire `lock1`.


```csharp
using System;
using System.Threading;

class Deadlock
{
    private readonly Object lock1 = new Object();
    private readonly Object lock2 = new Object();

    public void thread1()
    {
        lock (lock1)
        {
            Console.Out.WriteLine("Thread 1 acquired lock1");
            Thread.Sleep(10);
            Console.Out.WriteLine("Thread 1 waiting on lock2");
            lock (lock2)    // Deadlock here
            {
            }
        }
    }

    public void thread2()
    {
        lock (lock2)
        {
            Console.Out.WriteLine("Thread 2 acquired lock2");
            Thread.Sleep(10);
            Console.Out.WriteLine("Thread 2 waiting on lock1");
            lock (lock1)    // Deadlock here
            {
            }
        }
    }
}

```
This problem is resolved by reordering the `lock` variables as shown below.


```csharp
using System;
using System.Threading;

class DeadlockFixed
{
    private readonly Object lock1 = new Object();
    private readonly Object lock2 = new Object();

    public void thread1()
    {
        lock (lock1)
        {
            Console.Out.WriteLine("Thread 1 acquired lock1");
            Thread.Sleep(10);
            Console.Out.WriteLine("Thread 1 waiting on lock2");
            lock (lock2)
            {
            }
        }
    }

    public void thread2()
    {
        lock (lock1)    // Fixed
        {
            Console.Out.WriteLine("Thread 2 acquired lock1");
            Thread.Sleep(10);
            Console.Out.WriteLine("Thread 2 waiting on lock2");
            lock (lock2)    // Fixed
            {
            }
        }
    }
}

```

## References
* MSDN, C\# Reference: [lock Statement](http://msdn.microsoft.com/en-us/library/c5kehkcz%28v=vs.110%29.aspx).
* The CERT Oracle Coding Standard for Java: [ LCK07-J. Avoid deadlock by requesting and releasing locks in the same order](https://www.securecoding.cert.org/confluence/display/java/LCK07-J.+Avoid+deadlock+by+requesting+and+releasing+locks+in+the+same+order).
* Common Weakness Enumeration: [CWE-662](https://cwe.mitre.org/data/definitions/662.html).
