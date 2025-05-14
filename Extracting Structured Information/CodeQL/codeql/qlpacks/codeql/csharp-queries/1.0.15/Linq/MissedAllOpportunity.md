# Missed opportunity to use All
Often a programmer wants to check that all the elements of a given sequence satisfy some predicate. A common pattern for this is to create a flag and then iterate over the sequence, changing the flag and breaking out of the loop if the element being examined does not satisfy the predicate.


## Recommendation
This pattern works well and is also available as the `All` method in LINQ. It is better to use a library method in preference to writing your own pattern unless you have a specific need for a custom version. In particular, this makes the code easier to read (the intent is more clearly expressed), shorter, less error-prone and more maintainable.


## Example
In this example the list is iterated in order to check if every element is even.


```csharp
class MissedAllOpportunity
{
    public static void Main(string[] args)
    {
        List<int> lst = new List<int> { 2, 4, 18, 12, 80 };

        bool allEven = true;
        foreach (int i in lst)
        {
            if (i % 2 != 0)
            {
                allEven = false;
                break;
            }
        }
        Console.WriteLine("All Even = " + allEven);
    }
}

```
The LINQ `All` method can be used to accomplish this in a much simpler fashion.


```csharp
class MissedAllOpportunityFix
{
    public static void Main(string[] args)
    {
        List<int> lst = new List<int> { 2, 4, 18, 12, 80 };

        Console.WriteLine("All Even = " + lst.All(i => i % 2 == 0));
    }
}

```

## References
* MSDN: [Enumerable.All&lt;TSource&gt; Method](http://msdn.microsoft.com/en-us/library/bb548541.aspx).
