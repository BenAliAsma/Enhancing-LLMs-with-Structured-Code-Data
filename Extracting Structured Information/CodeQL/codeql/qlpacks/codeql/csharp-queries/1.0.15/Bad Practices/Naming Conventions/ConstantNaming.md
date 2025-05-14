# Public static read-only fields should be named in PascalCase
Constants not named with PascalCase can be confusing to other developers.


## Recommendation
As per Microsoft's suggested naming conventions, all constant fields should be named using PascalCase.


## Example
This example demonstrates several different capitalizations that trigger alerts for this rule. Another developer working on the program might not realize they refer to constants. The ` PascalCase` constant is the only one named correctly.


```csharp
class ConstantNaming
{
    public static readonly int UPPERCASE = 23;
    public static readonly int camelCase = 9;
    public static readonly int lowercase = 2;

    public static readonly int PascalCase = 121;
}

```

## References
* MSDN: [Capitalization Conventions](http://msdn.microsoft.com/en-gb/library/vstudio/ms229043%28v=vs.100%29.aspx).
