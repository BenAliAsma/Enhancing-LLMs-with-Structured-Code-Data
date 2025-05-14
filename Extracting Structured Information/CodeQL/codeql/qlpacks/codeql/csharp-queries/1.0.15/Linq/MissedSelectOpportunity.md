# Missed opportunity to use Select
It is common to see loops that immediately compute a value from their iteration variable and then never use the iteration variable again in the rest of the loop (see example below). The intent of such loops is arguably not to iterate over the original sequence at all, but to iterate over the sequence that results from transforming the original sequence in some manner.


## Recommendation
There is a good case to be made that the code is more readable if this intent is expressed explicitly, which can be done by using LINQ to perform a `Select` on the input sequence. The resulting code is clearer due to better separation of concerns.


## Example
This example iterates over a list of i<sup>2</sup>.


```csharp
class MissedSelectOpportunity
{
    public static void Main(string[] args)
    {
        List<int> lst = Enumerable.Range(1, 5).ToList();

        foreach (int i in lst)
        {
            int j = i * i;
            Console.WriteLine(j);
        }
    }
}

```
This could be better expressed by using LINQ's `Select` method with a lambda expression.


```csharp
class MissedSelectOpportunityFix
{
    public static void Main(string[] args)
    {
        List<int> lst = Enumerable.Range(1, 5).ToList();

        foreach (int j in lst.Select(i => i * i))
        {
            Console.WriteLine(j);
        }
    }
}

```

## References
* MSDN: [Enumerable.Select Method](http://msdn.microsoft.com/en-us/library/system.linq.enumerable.select.aspx).
