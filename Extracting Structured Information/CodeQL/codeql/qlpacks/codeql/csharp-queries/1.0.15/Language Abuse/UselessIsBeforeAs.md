# Useless 'is' before 'as'
The C\# `as` operator is equivalent to testing if an object is of a specific type with `is` and then performing a cast or returning null depending on the result. Performing an `is` check before calling `as` is completely redundant because an `is ` check is performed twice.


## Recommendation
Just use `as` on its own and then perform a null check on the result to determine if the cast was successful. Note in passing that it is no better to replace the 'as' with a cast, since that also does a type test internally.


## Example
In this example two type checks are performed for `a` against `Rectangle`.


```csharp
class UselessIsBeforeAs
{
    class Square
    {
        public int width;
    }
    class Rectangle : Square
    {
        public int height;
    }
    public static void Main(string[] args)
    {
        Square a = new Rectangle();
        if (a is Rectangle)
        {
            Rectangle r = a as Rectangle;
            Console.WriteLine(r.height);
        }
    }
}

```
Here is the same function performed more efficiently by omitting the extra type check.


```csharp
class UselessIsBeforeAs
{
    class Square
    {
        public int width;
    }
    class Rectangle : Square
    {
        public int height;
    }
    public static void Main(string[] args)
    {
        Square a = new Rectangle();
        Rectangle r = a as Rectangle;
        if (r != null)
        {
            Console.WriteLine(r.height);
        }
    }
}

```

## References
* MSDN: [is](http://msdn.microsoft.com/en-us/library/scekt9xw(v=vs.71).aspx)
* MSDN: [as](http://msdn.microsoft.com/en-us/library/cscsdfbt(v=vs.71).aspx)
* Common Weakness Enumeration: [CWE-561](https://cwe.mitre.org/data/definitions/561.html).
