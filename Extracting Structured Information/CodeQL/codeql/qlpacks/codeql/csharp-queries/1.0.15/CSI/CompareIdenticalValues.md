# Comparison of identical values
If two identical expressions are compared (that is, checked for equality or inequality), this is typically an indication of a mistake, because the Boolean value of the comparison is always the same. Often, it indicates that the wrong qualifier has been used on a field access.

An exception applies to inequality (`!=`) and equality (`==`) tests of a floating point variable with itself: the special floating point value `NaN` ("not-a-number") is the only value that is not considered to be equal to itself. Thus, the test `x != x` where `x` is a `float` or `double` variable is equivalent to checking whether `x` is `NaN`, and similarly for `x == x`.


## Recommendation
It is never good practice to compare a value with itself. If you require constant behavior, use the Boolean literals `true` and `false`, rather than encoding them obscurely as `1 == 1` or similar.

If an inequality test (using `!=`) of a floating point variable with itself is intentional, it should be replaced by `double.IsNaN(...)` or `float.IsNaN(...)` for readability. Similarly, if an equality test (using `==`) of a floating point variable with itself is intentional, it should be replaced by `!double.IsNaN(...)` or `!float.IsNaN(...)`.


## Example
In this example the developer clearly meant to compare age with `personObj.age` but instead compared age with itself.


```csharp
class Person
{
    private string name;
    private int age;
    public Person(string name, int age)
    {
        this.name = name;
        this.age = age;
    }
    public override bool Equals(object obj)
    {
        Person personObj = obj as Person;
        if (personObj == null)
        {
            return false;
        }
        return name == personObj.name && age == age; // BAD
    }
}

```

## References
* MSDN, C\# Reference, [Compiler Warning (level 3) CS1718](http://msdn.microsoft.com/en-GB/library/78kc05h3(v=vs.90).aspx).
* MSDN, C\# Reference, [Single.NaN Field](https://msdn.microsoft.com/en-us/library/system.single.nan(v=vs.110).aspx).
* MSDN, C\# Reference, [Double.NaN Field](https://msdn.microsoft.com/en-us/library/system.double.nan(v=vs.110).aspx).
