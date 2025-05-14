# Poor error handling: catch of NullReferenceException
Catching `NullReferenceException` should not be used as an alternative to checks and assertions for preventing dereferencing a null pointer.


## Recommendation
Check if the variable is null before dereferencing it.


## Example
The following example class, `findPerson` returns null if the person is not found.


```csharp
class CatchOfNullReferenceException
{
    public static Person findPerson(string name)
    {
        // ...
    }

    public static void Main(string[] args)
    {
        Console.WriteLine("Enter name of person:");
        Person p = findPerson(Console.ReadLine());
        try
        {
            Console.WriteLine("Person is {0:D} years old", p.getAge());
        }
        catch (NullReferenceException e)
        {
            Console.WriteLine("Person not found.");
        }
    }
}

```
The following example has been updated to ensure that any null return values are handled correctly.


```csharp
class CatchOfNullReferenceExceptionFix
{
    public static Person findPerson(string name)
    {
        // ...
    }

    public static void Main(string[] args)
    {
        Console.WriteLine("Enter name of person:");
        Person p = findPerson(Console.ReadLine());
        if (p != null)
        {
            Console.WriteLine("Person is {0:D} years old", p.getAge());
        }
        else
        {
            Console.WriteLine("Person not found.");
        }
    }
}

```

## References
* Common Weakness Enumeration: [CWE-395](https://cwe.mitre.org/data/definitions/395.html).
