# Property value is not used when setting a property
Ignoring the value assigned to a property causes confusion and can lead to unexpected results. A user of the class would expect different results depending on the value assigned to the property, but when the value is ignored, then this does not happen.

There are generally three situations where this occurs:

1. The property is not intended to be assigned to. In this case, the caller expects the assignment to have some effect, but it does not.
1. The property represents a particular state using a `bool`. In this case, the caller might expect that setting the property to `false` would have a different effect than setting it to `true`.
1. The property is virtual or overridden. In this case, it is possible for some of the derived classes to use the property, but other classes to ignore it. For this reason, virtual and overridden properties are excluded from the results.

## Recommendation
Empty setters (`set` with an empty body) should simply be removed.

Where the setter does not use the assigned value, replace `set` with a method that does not take a parameter. Alternatively, change the implementation of `set` to use the assigned value, or to throw an exception if the assigned value is invalid.


## Example
The following example shows two cases where `set` ignores the assigned value. In the first case, the property does not support `set`, so the implementation is empty. In the second case, the `set` code has the same effect for all assigned values.


```csharp
class Window
{
    const int screenWidth = 1280;
    int x0, x1;

    public int Width
    {
        get { return x1 - x0; }

        // BAD: Setter has no effect
        set { }
    }

    public bool FullWidth
    {
        get { return x0 == 0 && x1 == screenWidth; }

        // BAD: This is confusing if value==false
        set { x0 = 0; x1 = screenWidth; }
    }
}

```
The first of these problems can be fixed by simply removing the empty `set`. The second problem can be fixed by implementing the functionality in a method instead of in a property.


```csharp
class Window
{
    const int screenWidth = 1280;
    int x0, x1;

    public int Width
    {
        get { return x1 - x0; }
    }

    public bool IsFullWidth
    {
        get { return x0 == 0 && x1 == screenWidth; }
    }

    public void MakeFullWidth()
    {
        x0 = 0; x1 = screenWidth;
    }
}

```

## References
* MSDN, C\# Programming Guide: [Using Properties](http://msdn.microsoft.com/en-us/library/w86s7x04.aspx).
