# Missing documentation for public class or method
Public APIs (including methods, constructors, classes, structs and interfaces) should be documented to help make the API easier to understand and maintain. For the purpose of code maintainability, it is also advisable to document non-public APIs.


## Recommendation
Add appropriate documentation. The documentation comment should describe *what* the method or constructor does rather than *how* it does it, to allow for any potential implementation change that is invisible to users of an API. It should include the following:

* A `<summary>` tag giving a brief description of the class or method
* `<param>` tags to describe all arguments to methods and constructors
* A `<returns>` tag if the method returns a value
* `<exception>` tags for all exceptions thrown by the method
* `<typeparam>` tags for type parameters to classes and methods
* A description of any preconditions or postconditions
* Any other important aspects such as side-effects and thread safety
Documentation for users of an API should be written using the standard documentation format. This can be accessed conveniently by users of an API from within standard IDEs, and can be transformed automatically into HTML format.


## Example
The following example shows a fully documented class, illustrating the use of `<summary>`, `<param>`, `<returns>`, and `<exception>` tags.


```csharp
using System;
using System.Threading;

/// <summary>
///   A minimal threadsafe counter.
/// </summary>
class AtomicCounter
{
    /// <summary>
    ///   The current value of the counter.
    /// </summary>
    int currentValue = 0;

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
}

```

## References
* MSDN, C\# Programming Guide: [XML Documentation Comments](http://msdn.microsoft.com/en-us/library/b2s063f7.aspx), [How to: Use the XML Documentation Features](http://msdn.microsoft.com/en-us/library/z04awywx.aspx), [Recommended Tags for Documentation Comments](http://msdn.microsoft.com/en-us/library/5ast78ax.aspx).
