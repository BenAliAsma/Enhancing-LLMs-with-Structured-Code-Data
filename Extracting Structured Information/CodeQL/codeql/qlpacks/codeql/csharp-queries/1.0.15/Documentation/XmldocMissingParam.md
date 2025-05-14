# Missing documentation for parameter
A public method or constructor that does not have a `<param>` tag for each parameter makes an API more difficult to understand and maintain.


## Recommendation
The documentation comment for a method or constructor should include `<param>` tags that describe each parameter.


## Example
The following example shows a good documentation comment, which clearly explains the method's parameter.


```csharp
/// <summary>
///   Increments the value of the counter.
/// </summary>
///
/// <param name="incrementBy">The amount to increment.</param>
/// <exception cref="System.OverflowException">If the counter would overflow.</exception>
/// <returns>The new value of the counter.</returns>
///
/// <remarks>This method is threadsafe.</remarks>
public int Increment(int incrementBy = 1)
{
    int oldValue, newValue;
    do
    {
        oldValue = currentValue;
        newValue = oldValue + incrementBy;
        if (newValue < 0) throw new OverflowException("Counter value is out of range");
    }
    while (oldValue != Interlocked.CompareExchange(ref currentValue, newValue, oldValue));
    return newValue;
}

```

## References
* MSDN, C\# Programming Guide: [XML Documentation Comments](http://msdn.microsoft.com/en-us/library/b2s063f7.aspx), [How to: Use the XML Documentation Features](http://msdn.microsoft.com/en-us/library/z04awywx.aspx), [Recommended Tags for Documentation Comments](http://msdn.microsoft.com/en-us/library/5ast78ax.aspx).
