# Cast from abstract to concrete collection
Casting from an abstract collection to a concrete implementation is bad practice. It makes your code fragile because it becomes more difficult to change which implementation you are using at a later date.


## Recommendation
Consider using the abstract collection's methods and remove the cast.


## Example
The example shows casting from an `IEnumerable<string>` to a `List<string>`. This should be avoided where possible.


```csharp
using System.Collections.Generic;

class Bad
{
    public static void Main(string[] args)
    {
        var names = GetNames();
        var list = (List<string>) names;
        list.Add("Eve");
    }

    static IEnumerable<string> GetNames()
    {
        var ret = new List<string>()
        {
            "Alice",
            "Bob"
        };
        return ret;
    }
}

```

## References
* C\# Corner, [C\# Interface Based Development](https://www.c-sharpcorner.com/article/C-Sharp-interface-based-development/).
* Common Weakness Enumeration: [CWE-485](https://cwe.mitre.org/data/definitions/485.html).
