# Bad multiple iteration
A C\# `IEnumerable` can be thought of as a lazily-yielded sequence of elements. Although an `IEnumerable` will often be backed by a `List`, or some other collection that can be iterated over multiple times, it is not guaranteed that the referred-to sequence will always be repeatable - if we iterate over an `IEnumerable` multiple times, it is perfectly possible that we will get different results each time.


## Recommendation
If you need to iterate over a sequence multiple times, you must make sure that it is repeatable, and you should change the type of the sequence to make sure that this constraint is enforced. A common way to do this (although it is not applicable in every situation) is to call LINQ's ` ToList` method on your sequence at the time you create it and then assign it to an `IList ` with the appropriate element type. Since it is safe to iterate over a list multiple times, this solves the problem. However, you must be careful when doing this - in the case of something like `File.ReadLines`, the sequence returned is deliberately lazy (there may not be enough heap space to keep the entire file in memory at once), so calling `ToList` would create other problems. In that situation, you may need to redesign the way your code works to keep the lazy behavior whilst avoiding multiple iteration.

As a concrete example of this, consider the situation where you need to check whether or not a lazy sequence contains any elements before zipping it with another sequence - in such a situation, an appropriate technique would be to write a wrapper that tries to consume an element of the sequence (to check whether or not it is empty), and then either returns a sequence that yields the consumed element followed by the rest of the sequence or returns null. By checking the result against null, it is then possible to test whether or not the original sequence was empty; if it was not empty, the new sequence can be used to yield all of the elements of the original sequence as desired.


## Example
For example, the method `NonRepeatable` in the example below yields a non-repeatable sequence of three integers. The second loop over `nr` does nothing, because the non-repeatable sequence has already been consumed. Although this example is somewhat contrived, non-repeatable sequences are unfortunately not all that rare - in particular, C\# library methods such as `File.ReadLines` return such sequences. As a result, it is not safe to iterate over an unknown `IEnumerable` multiple times, because you cannot know for certain that it refers to a repeatable sequence.


```csharp
class BadMultipleIteration
{
    private static int count = 1;

    private static IEnumerable<int> NonRepeatable()
    {
        for (; count <= 3; count++)
        {
            yield return count;
        }
    }

    public static void Main(string[] args)
    {
        IEnumerable<int> nr = NonRepeatable();
        foreach (int i in nr)
            Console.WriteLine(i);

        foreach (int i in nr)
            Console.WriteLine(i);
    }
}

```
This example can be made to work as expected by storing the result of the enumeration method as a list.


```csharp
class BadMultipleIterationFix
{
    private static int count = 1;
    private static IEnumerable<int> NonRepeatable()
    {
        for (; count <= 3; count++)
        {
            yield return count;
        }
    }

    public static void Main(string[] args)
    {
        IList<int> nr = NonRepeatable().ToList<int>();
        foreach (int i in nr)
            Console.WriteLine(i);

        foreach (int i in nr)
            Console.WriteLine(i);
    }
}

```

## References
* MSDN: [IEnumerable Interface](http://msdn.microsoft.com/en-us/library/system.collections.ienumerable.aspx).
* Common Weakness Enumeration: [CWE-834](https://cwe.mitre.org/data/definitions/834.html).
