# Incorrect type parameter name in documentation
Documentation comments for public classes and methods should use the `<typeparam>` tag to describe the available type parameters. If the comment includes any empty, incorrect or outdated parameter names then this will make the documentation more difficult to read.


## Recommendation
The documentation comment for a class or method should always use non-empty `<typeparam>` tags that match actual parameter names.


## Example
The following example shows some examples of correct and incorrect ways to document type parameters. parameter.


```csharp
/// <summary>
///   BAD: An example of a class that lacks documentation
///   for a type parameter.
/// </summary>
class Stack<T>
{
}

/// <summary>
///   BAD: An example of a class that has incorrect documentation
///   for the type parameter.
/// </summary>
/// <typeparam name="X">The type of each stack element.</typeparam>
class Stack<T>
{
}

/// <summary>
///   GOOD: An example of a class whose type parameters are
///   correctly documented.
/// </summary>
/// <typeparam name="T">The type of each stack element.</typeparam>
class Stack<T>
{
}

```

## References
* MSDN, C\# Programming Guide: [XML Documentation Comments](http://msdn.microsoft.com/en-us/library/b2s063f7.aspx), [How to: Use the XML Documentation Features](http://msdn.microsoft.com/en-us/library/z04awywx.aspx), [Recommended Tags for Documentation Comments](http://msdn.microsoft.com/en-us/library/5ast78ax.aspx).
