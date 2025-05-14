# Unused field
This rule finds private fields that seem to never be used. Dead fields are often deprecated pieces of code, and should be removed as they may increase object code size, decrease code maintainability, and create the possibility of misuse. This query not only spots private fields that are not directly accessed, but also private fields that are solely accessed from methods which are themselves dead.


## Recommendation
Carefully check the class for errors. If there are none then remove the unused field. It may be necessary to consider the side effects of any methods called during the initialization of dead field.


## Example
In the first example the name field is not used at all due to a mistake by the programmer. In the second example the field is only used from a dead method.


```csharp
class UnusedField1
{
    string name;  // BAD
    string email;

    public string GetName()
    {
        return email;
    }
    public string GetEmail()
    {
        return email;
    }
}

class UnusedField2
{
    string name; // BAD

    private string GetName()
    {
        return name;
    }
}

```
