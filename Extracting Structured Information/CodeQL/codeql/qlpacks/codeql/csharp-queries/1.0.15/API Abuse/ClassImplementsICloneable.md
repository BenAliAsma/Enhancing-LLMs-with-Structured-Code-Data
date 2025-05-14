# Class implements ICloneable
The purpose of implementing the `ICloneable` interface is to advertise that instances of your class can be cloned, but there are a number of good reasons why doing so is generally a bad idea.

First, the semantics of `ICloneable.Clone()` are not well-specified. The documentation states that it "creates a new object that is a copy of the current instance", but it then fails to specify what is meant by "copy". This poses a problem, because there are two general ways in which an object can be copied, either deeply or shallowly (a deep copy clones an object and all the objects it references recursively, whereas a shallow copy only clones the top of the object graph - see example below). Since `ICloneable` does not specify which type of copying should be performed, different implementing classes tend to pick one or the other arbitrarily, making it impossible to call `Clone()` with any certainty in general about what will happen - the method is thus useless.

A further problem with `ICloneable` is that if a class implements it, all of its subtypes must implement it as well, as must all of the types of its members if a deep copy is being implemented. Not only does this make using `ICloneable` rather viral, but also rather error-prone, because it is very easy to forget to override `Clone()` in subtypes. As such, using `ICloneable` is widely discouraged.


## Recommendation
Define your own `Clone()` or `Copy()` methods and document whether they perform deep or shallow copying.


## Example

```csharp
using System;

class Bad
{
    class Thing
    {
        public int I { get; set; }
        public Thing(int i) { I = i; }
    }

    class Shallow : ICloneable
    {
        public Thing T { get; set; }
        public Shallow(Thing t) { T = t; }

        // Implements a shallow clone (compliant with the spec)
        public object Clone() { return new Shallow(T); }
    }

    class Deep : ICloneable
    {
        public Thing T { get; set; }
        public Deep(Thing t) { T = t; }

        // Implements a deep clone (also compliant with the spec)
        public object Clone() { return new Deep(new Thing(T.I)); }
    }

    static void Main(string[] args)
    {
        var s1 = new Shallow(new Thing(23));
        var s2 = (Shallow)s1.Clone();
        Console.WriteLine(s2.T.I); // 23
        s1.T.I = 9;
        Console.WriteLine(s2.T.I); // 9

        var d1 = new Deep(new Thing(23));
        var d2 = (Deep)d1.Clone();
        Console.WriteLine(d2.T.I); // 23
        d1.T.I = 9;
        Console.WriteLine(d2.T.I); // 23
    }
}

```

## References
* MSDN, [ICloneable Interface](https://msdn.microsoft.com/en-us/library/system.icloneable.aspx).
* B Wagner, *Effective C\#: 50 Specific Ways to Improve Your C\#* (Second Edition), Item 32: Avoid ICloneable.
