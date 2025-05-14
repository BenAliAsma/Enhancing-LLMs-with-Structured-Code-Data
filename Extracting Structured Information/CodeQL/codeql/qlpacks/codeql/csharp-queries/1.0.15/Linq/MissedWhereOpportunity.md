# Missed opportunity to use Where
Programmers sometimes need to iterative over a filtered version of a sequence, rather than the sequence itself. For example, you might want to print out only the numbers in the range \[1,10\] that are even. One standard way of doing this is to write a loop that iterates over the whole sequence, testing the variable each iteration to determine whether or not it is even. This is often written using either `if(!condition(var)) continue;` as the initial statement in the loop, or by enclosing the entire loop body with `if(condition(var))`.


## Recommendation
This pattern works well and is also available as the `Where` method in LINQ in C\# 3.5 and above. It is better to use a library method in preference to writing your own pattern unless you have a specific need for a custom version. In particular, this makes the code easier to read by expressing the intent better and by reducing the nesting depth of the code.


## Example
This example shows two ways of iterating over a series of integers and only performing an action on the even ones.


```csharp
class MissedWhereOpportunity
{
    public static void Main(string[] args)
    {
        List<int> lst = Enumerable.Range(1, 10).ToList();

        foreach (int i in lst)
        {
            if (i % 2 != 0)
                continue;
            Console.WriteLine(i);
            Console.WriteLine((i / 2));
        }

        foreach (int i in lst)
        {
            if (i % 2 == 0)
            {
                Console.WriteLine(i);
                Console.WriteLine((i / 2));
            }
        }
    }
}

```
This is far better expressed using the `Where` method.


```csharp
class MissedWhereOpportunityFix
{
    public static void Main(string[] args)
    {
        List<int> lst = Enumerable.Range(1, 10).ToList();

        foreach (int i in lst.Where(e => e % 2 == 0))
        {
            Console.WriteLine(i);
            Console.WriteLine((i / 2));
        }
    }
}

```

## References
* MSDN: [Enumerable.Where Method](http://msdn.microsoft.com/en-us/library/system.linq.enumerable.where.aspx).
