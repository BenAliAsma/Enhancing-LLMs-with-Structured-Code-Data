# Array constant vulnerable to change
Arrays might be made `static readonly` to prevent their contents from being changed. This doesn't have the desired effect because arrays are mutable. The `readonly` modifier prevents the array from being replaced by a new array but it does not prevent the contents of the array from being changed.


## Recommendation
Consider whether the array could be split up into separate constants. If the array cannot be split then you may wish to use a `ReadOnlyCollection` instead of an array.


## Example
In this example the `Foo` array is `readonly` but it is still modified by the `Main` method.


```csharp
class Bad
{
    public static readonly string[] Foo = { "hello", "world" };
    public static void Main(string[] args)
    {
        Foo[0] = "goodbye";
    }
}

```
This example uses a `ReadOnlyCollection`. Any attempt to modify `Foo` will cause the program not to compile.


```csharp
using System.Collections.ObjectModel;

class Good
{
    public static readonly ReadOnlyCollection<string> Foo
        = new ReadOnlyCollection<string>(new string[] { "hello", "world" });
    public static void Main(string[] args)
    {

    }
}

```

## References
* MSDN, C\# Programming Guide, [Arrays as Objects](https://docs.microsoft.com/en-us/dotnet/csharp/programming-guide/arrays/arrays-as-objects).
* MSDN, [ReadOnlyCollection&lt;T&gt;](http://msdn.microsoft.com/en-us/library/ms132474.aspx).
* Common Weakness Enumeration: [CWE-582](https://cwe.mitre.org/data/definitions/582.html).
