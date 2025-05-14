# Missing a summary in documentation comment
Documentation comments should contain a `<summary>` tag which briefly describes the purpose of the class or method.


## Recommendation
Add a `<summary>` tag to any documentation comments which do not have one.


## Example
The following example shows a `<summary>` tag describing a class.


```csharp
/// <summary>
///   A minimal threadsafe counter.
/// </summary>
class AtomicCounter
{
    // ...
}

```

## References
* MSDN, C\# Programming Guide: [XML Documentation Comments](http://msdn.microsoft.com/en-us/library/b2s063f7.aspx), [How to: Use the XML Documentation Features](http://msdn.microsoft.com/en-us/library/z04awywx.aspx), [Recommended Tags for Documentation Comments](http://msdn.microsoft.com/en-us/library/5ast78ax.aspx).
