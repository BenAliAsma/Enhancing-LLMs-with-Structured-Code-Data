# Missing documentation for exception
A public method or constructor that throws an exception but does not have a documentation tag for the exception makes an API more difficult to understand and maintain.


## Recommendation
The documentation comment for a method or constructor should include an `<exception>` tag that describes each thrown exception, and the `cref` attribute of the tag should match the type of the exception that is thrown.


## Example
The following example shows a good documentation comment, which clearly explains the method's thrown exception.


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
