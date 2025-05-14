# Field is never assigned a non-default value
Fields are automatically given a default value when declared. If they are never assigned then they will always have their default value.


## Recommendation
Check if an assignment to the field has been accidentally omitted. If it hasn't then consider removing the variable or explicitly initializing the field to its default value.


## Example
In this example the programmer has forgotten to initialize the "name" field. This is the kind of mistake that this rule can detect.


```csharp
class Person
{
    private string name;
    private int age;
    public Person(string name, int age)
    {
        this.age = age;
    }
}

```

## References
* MSDN, C\# Reference, [Compiler Warning (level 4) CS0649](http://msdn.microsoft.com/en-us/library/03b5270t(v=vs.71).aspx).
* Common Weakness Enumeration: [CWE-457](https://cwe.mitre.org/data/definitions/457.html).
