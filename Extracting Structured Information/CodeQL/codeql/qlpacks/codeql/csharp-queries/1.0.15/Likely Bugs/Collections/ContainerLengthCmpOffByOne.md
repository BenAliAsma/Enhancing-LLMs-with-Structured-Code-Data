# Off-by-one comparison against container length
An indexing operation against a collection, string or array should use an index at most one less than the length. If the index to be accessed is compared to be less than or equal to the length (`<=`), instead of less than the length (`<`), the index could be out of bounds.


## Recommendation
Use less than (`<`) rather than less than or equals (`<=`) when comparing a potential index against a length. For loops that iterate over every element in an array or collection, prefer to use the `foreach` syntax instead of looping over explicit indexes.


## Example
The following example shows two methods which identify whether a value appears in a comma-separated list of values.

In the first example, a loop using an index variable - `i` - is used to iterate over the elements in the comma-separated list. However, the terminating condition of the loop is incorrectly specified as `i <= values.Length`. This condition will succeed if `i` is equal to `values.Length`, but the access `values[i]` in the body of the loop will fail because the index is out of bounds.

One potential solution would be to replace `i <= values.Length` with `i < values.Length`. However, arrays in C\# can also be used in `foreach` loops. This is shown in the second example. This circumvents the need to specify an index at all, and therefore prevents errors where the index is out of bounds.


```csharp
using System.IO;
using System;

class ContainerLengthOffByOne
{
    public static boolean BadContains(string searchName, string names)
    {
        string[] values = names.Split(',');
        // BAD: index could be equal to length
        for (int i = 0; i <= values.Length; i++)
        {
            // When i = length, this access will be out of bounds
            if (values[i] == searchName)
            {
                return true;
            }
        }
    }

    public static boolean GoodContains(string searchName, string names)
    {
        string[] values = names.Split(',');
        // GOOD: Avoid using indexes, and use foreach instead
        foreach (string name in values)
        {
            if (name == searchName)
            {
                return true;
            }
        }
    }
}

```

## References
* Common Weakness Enumeration: [CWE-193](https://cwe.mitre.org/data/definitions/193.html).
