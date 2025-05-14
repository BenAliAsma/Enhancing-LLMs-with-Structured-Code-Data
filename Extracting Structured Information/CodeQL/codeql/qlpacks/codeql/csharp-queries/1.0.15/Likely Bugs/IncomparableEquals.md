# Equals on incomparable types
Calling `x.Equals(y)` on incomparable types will almost always return false. If two classes do not have a common parent class their instances are considered incomparable.


## Recommendation
Carefully check the code for errors.


## Example
In this example both calls to the Equals method will always return false regardless of the contents of the `ArrayList` or `String` because `ArrayList`s and `String`s are incomparable.


```csharp
using System.Collections;

class IncomparableEquals
{
    public static void Main(string[] args)
    {
        ArrayList apple = new ArrayList();
        String orange = "foo";
        Console.WriteLine(apple.Equals(orange)); // BAD
        Console.WriteLine(orange.Equals(apple)); // BAD
    }
}

```

## References
* MSDN, [Object.Equals Method](http://msdn.microsoft.com/en-us/library/bsc2ak47.aspx).
