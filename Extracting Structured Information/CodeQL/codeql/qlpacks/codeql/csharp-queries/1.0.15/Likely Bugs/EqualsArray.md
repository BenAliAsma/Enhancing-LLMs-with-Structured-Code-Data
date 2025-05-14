# Equals on collections
There are some circumstances where you want to compare two arrays or collections using reference equality, but often you will want a deep comparison instead (that is, one that compares two collections on an element-by-element basis).


## Recommendation
If you intended to make a deep comparison then the solution depends on whether or not you are using C\# 3.5 or later. From C\# 3.5 onward LINQ is available and you can replace the call to ` Equals` with a call to the LINQ extension method `SequenceEqual`. If this is not possible then you can implement a helper function to compare arrays, which can then be used throughout your code.


## Example
This example outputs "False" because calling `Equals` on an array only does a reference comparison.


```csharp
class EqualsArray
{
    public static void Main(string[] args)
    {
        string[] strings = { "hello", "world" };
        string[] moreStrings = { "hello", "world" };
        Console.WriteLine(strings.Equals(moreStrings));
    }
}

```
The following example outputs "True" twice and uses two different methods of performing a deep comparison of the array.


```csharp
class EqualsArrayFix
{
    static bool DeepEquals<T>(T[] arr1, T[] arr2)
    {
        // If arr1 and arr2 refer to the same array, they are trivially equal.
        if (ReferenceEquals(arr1, arr2)) return true;

        // If either arr1 or arr2 is null and they are not both null (see the previous
        // check), they are not equal.
        if (arr1 == null || arr2 == null) return false;

        // If both arrays are non-null but have different lengths, they are not equal.
        if (arr1.Length != arr2.Length) return false;

        // Failing which, do an element-by-element compare.
        for (int i = 0; i < arr1.Length; ++i)
        {
            // Early out if we find corresponding array elements that are not equal.
            if (!arr1[i].Equals(arr2[i])) return false;
        }

        // If we get here, all of the corresponding array elements were equal, so the
        // arrays are equal.
        return true;
    }

    public static void Main(string[] args)
    {
        string[] strings = { "hello", "world" };
        string[] moreStrings = { "hello", "world" };
        Console.WriteLine(strings.SequenceEqual(moreStrings));
        Console.WriteLine(DeepEquals(strings, moreStrings));
    }
}

```

## References
* MSDN: [Enumerable.SequenceEqual Method](http://msdn.microsoft.com/en-us/library/system.linq.enumerable.sequenceequal.aspx).
