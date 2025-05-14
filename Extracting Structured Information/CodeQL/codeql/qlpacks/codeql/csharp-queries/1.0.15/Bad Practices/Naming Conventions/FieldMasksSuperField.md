# Field masks field in super class
This rule identifies a field that hides a field in a superclass because the field has the same name and, additionally, there are no methods in the subclass that refer to the field in the superclass directly using `super.field`. Redefining the field in the subclass is a common mistake.


## Recommendation
Check that the field should indeed be redefined in the subclass. If it should be redefined then consider changing its name to be more clear about what the field does.


## Example
In this example the "name" property in Employee masks the "name" property in Person. This was probably not what was intended.


```csharp
class Person
{
    private string name;
    public Person() { }
    public Person(string name)
    {
        this.name = name;
    }
}
class Employee : Person
{
    private string name; // BAD
    private string department;
    public Employee(string name, string department)
    {
        this.name = name;
        this.department = department;
    }
}

```

## References
* MSDN, C\# Programming Guide, [Hiding through inheritance](http://msdn.microsoft.com/en-us/library/aa691135(v=vs.71).aspx).
