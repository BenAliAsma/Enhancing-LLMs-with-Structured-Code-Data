# Number of properties
This metric counts the number of properties each reference type has. If a reference type has a large number of properties then it is possible it has too many responsibilities.


## Recommendation
If the type performs several different functions then it should be separated into different types. If the type only performs one function you may be able to group the properties into their own dedicated classes.


## Example
In this example the class has too many properties.


```csharp
class Person
{
    private string _firstName;
    private string _middleName;
    private string _lastName;

    public string firstName
    {
        get
        {
            return _firstName;
        }
        set
        {
            value = value.ToLower();
            char firstUC = char.ToUpper(value[0]);
            _firstName = firstUC + value.Substring(1);
        }
    }
    public string middleName
    {
        get
        {
            return _middleName;
        }
        set
        {
            value = value.ToLower();
            char firstUC = char.ToUpper(value[0]);
            _firstName = firstUC + value.Substring(1);
        }
    }
    public string lastName
    {
        get
        {
            return _lastName;
        }
        set
        {
            value = value.ToLower();
            char firstUC = char.ToUpper(value[0]);
            _firstName = firstUC + value.Substring(1);
        }
    }
    // more properties...
}

```

## Fix
The class only performs one function: representing a person. The properties can, however, be grouped together. There is an obvious connection between the properties associated with the person's name. As such the name properties could be extracted into a separate class as shown here.


```csharp
class Person
{
    private Name _name;
    public Name name
    {
        get
        {
            return name;
        }
        set
        {
            _name = value;
        }
    }
    // more fields and properties...
}
class Name
{
    private string _firstName;
    private string _middleName;
    private string _lastName;

    public string firstName
    {
        get
        {
            return _firstName;
        }
        set
        {
            value = value.ToLower();
            char firstUC = char.ToUpper(value[0]);
            _firstName = firstUC + value.Substring(1);
        }
    }
    public string middleName
    {
        get
        {
            return _middleName;
        }
        set
        {
            value = value.ToLower();
            char firstUC = char.ToUpper(value[0]);
            _firstName = firstUC + value.Substring(1);
        }
    }
    public string lastName
    {
        get
        {
            return _lastName;
        }
        set
        {
            value = value.ToLower();
            char firstUC = char.ToUpper(value[0]);
            _firstName = firstUC + value.Substring(1);
        }
    }
}

```

## References
* MSDN. C\# Programming Guide. [Properties](http://msdn.microsoft.com/en-us/library/x9fsa0sw(v=vs.80).aspx).
