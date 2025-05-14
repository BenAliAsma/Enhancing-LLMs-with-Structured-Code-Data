# Redundant Select
Passing an identity function to LINQ's `Select` method (either explicitly or implicitly) yields a sequence that is the same as the one on which `Select` was called - such a call is redundant.


## Recommendation
Remove the redundant select method call.


## Example
In this example the call to the `Select` method has no effect and can be removed.


```csharp
class RedundantSelect
{
    static void Main(string[] args)
    {
        List<int> lst = Enumerable.Range(1, 10).ToList();

        foreach (int i in lst.Select(e => e).Where(e => e % 2 == 0))
        {
            Console.WriteLine(i);
        }
    }
}

```
