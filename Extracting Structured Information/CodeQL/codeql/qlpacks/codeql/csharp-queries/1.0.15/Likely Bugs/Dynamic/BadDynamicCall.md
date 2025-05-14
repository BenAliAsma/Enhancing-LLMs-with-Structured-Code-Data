# Bad dynamic call
Method calls on variables declared with type 'dynamic' are resolved at runtime rather than compile-time - the actual type of the instance is determined, and an attempt is made to call a method on that type with the appropriate signature. If such a method does not exist, a ` RuntimeBinderException` is thrown.

This rule identifies calls to instances with the `dynamic` type where it can be statically determined that the call will throw a `RuntimeBinderException`.


## Recommendation
Ensure it is not possible to make a call to a dynamic instance of a type that lacks the appropriate method signature for handling that call.


## Example
In this example the program attempts to call `Foo` on a class that doesn't have a `Foo` method. This program is guaranteed to fail at runtime with a ` RuntimeBinderException`.


```csharp
class BadDynamicCall
{
    class WithFoo
    {
        public void Foo(int i) { }
    }

    class WithoutFoo { }

    public static void Main(string[] args)
    {
        dynamic o = new WithoutFoo();
        o.Foo(3);
    }
}

```

## References
* MSDN: [dynamic (C\# Reference)](http://msdn.microsoft.com/en-gb/library/dd264741.aspx).
* Common Weakness Enumeration: [CWE-628](https://cwe.mitre.org/data/definitions/628.html).
