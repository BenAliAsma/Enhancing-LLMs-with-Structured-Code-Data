# Pointless forwarding method
If a class contains two distinct methods of the same name such that:

* one of them is only ever called from the other
* and the calling method of the two only makes calls to the callee
* and the calling method makes no other calls
then the calling method is no more than a forwarding method for the callee. Forwarding methods are difficult to keep track of and can make public APIs more complicated.


## Recommendation
Merge the two methods.


## Example
In this example the `Print(string, string)` method is simply a forwarding method. Since `Print(string)` is not called anywhere else it serves no purpose.


```csharp
class PointlessForwardingMethod
{
    public static void Print(string firstName, string lastName)
    {
        Print(firstName + " " + lastName);
    }

    public static void Print(string fullName)
    {
        Console.WriteLine("Pointless forwarding methods are bad, " + fullName + "...");
    }

    public static void Main(string[] args)
    {
        Print("John", "Doe");
    }
}

```
This example could be easily improved by merging the two Print methods like so:


```csharp
class PointlessForwardingMethodFix
{
    public static void Print(string firstName, string lastName)
    {
        string fullName = firstName + " " + lastName;
        Console.WriteLine("Pointless forwarding methods are bad, " + fullName + "...");
    }

    public static void Main(string[] args)
    {
        Print("John", "Doe");
    }
}

```
