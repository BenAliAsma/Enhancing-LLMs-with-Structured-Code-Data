# Too many 'ref' parameters
Whilst passing method arguments using `ref` is occasionally a sensible thing to do (the canonical example is when writing a 'swap' method), overusing reference parameters can make methods needlessly difficult to understand and call.

Methods end up relying on their parameters being correctly initialized elsewhere and can have problems such as aliasing (when two parameters refer to the same object). They can be difficult to call because the references must refer to l-values (rather than temporary objects), so additional objects must be created to hold the results of the call. Their results can also be difficult to forward to other methods.


## Recommendation
Whilst it is not applicable in every situation, and some judgment must be applied, a common solution is to create a new class that wraps the values you are trying to set in the method and then modify the method to return a new instance of it.


## Example
In this example, populating the `name`, `address` and `tel` fields is done via a method that takes `ref` parameters. This is not very good practice and makes the method confusing.


```csharp
using System;

class Bad
{
    private static void PopulateDetails(ref string name, ref string address, ref string tel)
    {
        name = "Foo";
        address = "23 Bar Street";
        tel = "01234 567890";
    }

    private static void PrintDetails(string name, string address, string tel)
    {
        Console.WriteLine("Name: " + name);
        Console.WriteLine("Address: " + address);
        Console.WriteLine("Tel.: " + tel);
    }

    static void Main(string[] args)
    {
        string name = null, address = null, tel = null;
        PopulateDetails(ref name, ref address, ref tel);
        PrintDetails(name, address, tel);
    }
}

```
It is better to wrap the values in their own `Details` class and then return a new instance of `Details`. It is easier to pass to other functions correctly and is easier to understand.


```csharp
using System;

class Good
{
    class Details
    {
        public string Name { get; private set; }
        public string Address { get; private set; }
        public string Tel { get; private set; }

        public Details(string name, string address, string tel)
        {
            Name = name;
            Address = address;
            Tel = tel;
        }
    }

    private static Details PopulateDetails()
    {
        return new Details("Foo", "23 Bar Street", "01234 567890");
    }

    private static void PrintDetails(Details details)
    {
        Console.WriteLine("Name: " + details.Name);
        Console.WriteLine("Address: " + details.Address);
        Console.WriteLine("Tel.: " + details.Tel);
    }

    static void Main(string[] args)
    {
        PrintDetails(PopulateDetails());
    }
}

```
