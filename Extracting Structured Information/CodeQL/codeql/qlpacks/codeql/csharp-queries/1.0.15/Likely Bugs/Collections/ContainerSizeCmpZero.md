# Container size compared to zero.
A collection, string or array will always have size of at least zero. Checking that an object of one of these types has size greater than or equal to zero will always be true, while checking that it has size less than zero will always be false.


## Recommendation
For strings, it may be preferred to use the `System.String.IsNullOrEmpty` static method. In other cases, check that the `Count` or `Length` field is greater than (**not** equal to) zero.


## Example
The following example shows creation of a file guarded by comparison of a string length with zero. This can result in the attempted creation of a file with an empty name.


```csharp
using System.IO;
using System;

class ContainerSizeCmpZero
{
    private static FileStream MakeFile(String filename)
    {
        if (filename != null && filename.Length >= 0)
        {
            return File.Create(filename);
        }
        return File.Create("default.name");
    }
}

```
In the following revised example, the check against zero has been replaced with a call to `String.IsNullOrEmpty`. This correctly guards against the attempted creation of a file with an empty name.


```csharp
using System.IO;
using System;

class ContainerSizeCmpZero
{
    private static FileStream MakeFile(String filename)
    {
        if (filename != null && !(String.IsNullOrEmpty(filename)))
        {
            return File.Create(filename);
        }
        return File.Create("default.name");
    }
}

```

## References
* MSDN Library (.NET Framework 4.5): [String.IsNullOrEmpty Method](https://msdn.microsoft.com/en-us/library/system.string.isnullorempty(v=vs.110).aspx).
