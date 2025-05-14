# Inconsistently synchronized property
Data which is accessed concurrently from multiple threads is vulnerable to race conditions and other errors. `lock` statements are often needed to make concurrent code correct and ensure that the data is consistent. However `lock` statements must be used consistently on all methods which are potentially called concurrently.

When there is a `lock` statement on a property setter, it implies that the property could be assigned to concurrently, so the property could also be read concurrently. Therefore a `lock` statement is necessary on the property getter. Even simple getters require a `lock` statement to ensure that there is a memory barrier when reading a field.


## Recommendation
Examine the logic of the program to check whether the property could be read concurrently. Add a `lock` statement in the property getter if necessary.

Alternatively, remove the `lock` statement from the property setter if it is no longer needed.


## Example
This example shows a property setter which uses a `lock` statement, but there is no corresponding `lock` statement in the getter. This means that `count` is not synchronized with `GenerateDiagnostics()`, and there is a read barrier missing from the property getter, which could cause bugs on some CPU architectures.


```csharp
public int ErrorCount
{
    get
    {
        return count;
    }

    set
    {
        lock (mutex)
        {
            count = value;
            if (count > 0) GenerateDiagnostics();
        }
    }
}

```
The solution is to add a `lock` statement to the property getter, as follows:


```csharp
public int ErrorCount
{
    get
    {
        lock (mutex)
        {
            return count;
        }
    }

    set
    {
        lock (mutex)
        {
            count = value;
            if (count > 0) GenerateDiagnostics();
        }
    }
}

```

## References
* MSDN, C\# Reference: [lock Statement](http://msdn.microsoft.com/en-gb/library/c5kehkcz.aspx).
* Wikipedia: [Memory barrier](https://en.wikipedia.org/wiki/Memory_barrier).
* Common Weakness Enumeration: [CWE-662](https://cwe.mitre.org/data/definitions/662.html).
