# Use of default ToString()
Calling `System.Object`'s (or `System.ValueType`'s) `ToString` method on a value returns the fully qualified name of the type of that value. In most cases this is not useful, or what was intended. This rule finds explicit and implicit calls to the default `ToString` methods.


## Recommendation
Override the default `ToString` method, if possible, or perform bespoke string conversion.


## Example
In the following example, the default `ToString` method is invoked first on an object of type `Person`, and then on an integer array. The output results are `p: Bad+Person` and `ints: System.Int32[]`, respectively.


```csharp
using System;

class Bad
{
    static void Main(string[] args)
    {
        var p = new Person("Eric Arthur Blair");
        Console.WriteLine("p: " + p);

        var ints = new int[] { 1, 2, 3 };
        Console.WriteLine("ints: " + ints);
    }

    class Person
    {
        private string Name;

        public Person(string name)
        {
            this.Name = name;
        }
    }
}

```
In the fixed example, the `ToString` method is overridden in the class `Person`, and `string.Join` is used to print the elements of the integer array (it is not possible to override `ToString` in that case). The output results are `p: Eric Arthur Blair` and `ints: 1, 2, 3`, respectively.


```csharp
using System;

class Good
{
    static void Main(string[] args)
    {
        var p = new Person("Eric Arthur Blair");
        Console.WriteLine("p: " + p);

        var ints = new int[] { 1, 2, 3 };
        Console.WriteLine("ints: " + string.Join(", ", ints));
    }

    class Person
    {
        private string Name;

        public Person(string name)
        {
            this.Name = name;
        }

        public override string ToString() => Name;
    }
}

```

## References
* MSDN: [Object.ToString Method](http://msdn.microsoft.com/en-us/library/system.object.tostring.aspx), [ValueType.ToString Method](https://msdn.microsoft.com/en-us/library/wb77sz3h(v=vs.110).aspx).
