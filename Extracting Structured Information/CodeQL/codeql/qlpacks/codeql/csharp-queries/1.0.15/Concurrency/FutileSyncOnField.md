# Futile synchronization on field
A `lock` statement that locks a field which is modified is unlikely to provide thread safety. This is because different threads only lock the value of the field, not the field itself. Since the value of the field changes, different threads are locking different objects, and so are not mutually exclusive.


## Recommendation
Instead of locking the field itself, use a dedicated object for locking. The object should be `private` and `readonly` to ensure that it cannot be modified or locked elsewhere.


## Example
In the following example, the method `AddItem` can be called concurrently on different threads. `AddItem` attempts to protect the field `total` using a `lock` statement. However concurrent use of `AddItem` results in the value of `total` being incorrect.


```csharp
class Adder
{
    dynamic total = 0;

    public void AddItem(int item)
    {
        lock (total)     // Wrong
        {
            total = total + item;
        }
    }
}

```
The following code resolves this problem by using a dedicated object `mutex` for the lock.


```csharp
using System;

class Adder
{
    dynamic total = 0;

    private readonly Object mutex = new Object();

    public void AddItem(int item)
    {
        lock (mutex)    // Fixed
        {
            total = total + item;
        }
    }
}

```

## References
* MSDN, C\# Reference: [lock Statement](http://msdn.microsoft.com/en-us/library/c5kehkcz%28v=vs.110%29.aspx).
* Common Weakness Enumeration: [CWE-662](https://cwe.mitre.org/data/definitions/662.html).
* Common Weakness Enumeration: [CWE-366](https://cwe.mitre.org/data/definitions/366.html).
