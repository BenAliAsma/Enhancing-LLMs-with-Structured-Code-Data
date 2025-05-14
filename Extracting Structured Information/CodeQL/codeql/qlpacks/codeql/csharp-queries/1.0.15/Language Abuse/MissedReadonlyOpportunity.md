# Missed 'readonly' opportunity
A private field where all assignments occur as part of the declaration or in a constructor in the same class can be `readonly`. Making a field `readonly` prevents unintended assignments after object initialization.


## Recommendation
Add a `readonly` modifier to the field, unless changes to the field are allowed after object initialization.


## Example
In the following example, the field `Field` is only assigned to in the constructor, but it can still be modified after object initialization.


```csharp
class Bad
{
    int Field;

    public Bad(int i)
    {
        Field = i;
    }
}

```
In the revised example, the field is made `readonly`.


```csharp
class Good
{
    readonly int Field;

    public Good(int i)
    {
        Field = i;
    }
}

```

## References
* Microsoft: [readonly](https://docs.microsoft.com/en-us/dotnet/csharp/language-reference/keywords/readonly).
