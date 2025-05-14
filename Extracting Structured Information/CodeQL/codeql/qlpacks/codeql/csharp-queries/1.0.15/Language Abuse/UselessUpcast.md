# Useless upcast
In most situations, casting an expression where there exists a corresponding implicit conversion serves no purpose.


## Recommendation
Remove the redundant cast.


## Example
In this example, casting explicitly from `Sub` to `Super` is redundant.


```csharp
class Bad
{
    class Super {}
    class Sub : Super {}

    void M()
    {
        var sub = new Sub();
        Super super = (Super)sub;
    }
}

```
The code above can be fixed, either by removing the explicit cast, or by making `super` an implicitly typed (`var`) variable.


## References
* Microsoft: [Casting and Type Conversions](https://docs.microsoft.com/en-us/dotnet/csharp/programming-guide/types/casting-and-type-conversions).
* Common Weakness Enumeration: [CWE-561](https://cwe.mitre.org/data/definitions/561.html).
