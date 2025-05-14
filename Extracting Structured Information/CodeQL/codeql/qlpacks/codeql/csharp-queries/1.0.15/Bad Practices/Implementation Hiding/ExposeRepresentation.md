# Exposing internal representation
Sometimes a method of a class can expose an internal field to change by other code if it returns a reference to a mutable private field.


## Recommendation
There are several ways to address this problem depending on your situation. One of the best ways is to use an immutable object to store fields. References to this object can be passed outside the class but the objects are immutable so they cannot be changed.

Another good way of preventing external modification of private fields is to only ever return a copy of the object referred to by the field. This is called "defensive copying". If the copy is changed then internal fields will not be affected.


## Example
This example clearly demonstrates the problem with passing references to mutable objects outside a class. In this case it was possible to modify the values in the array despite the `Range` class not offering any method to do so.


```csharp
using System;

class Bad
{
    class Range
    {
        private int[] rarray = new int[2];

        public Range(int min, int max)
        {
            if (min <= max)
            {
                rarray[0] = min;
                rarray[1] = max;
            }
        }

        public int[] Get() => rarray;
    }

    public static void Main(string[] args)
    {
        var r = new Range(1, 10);
        var r_range = r.Get();
        r_range[0] = 500;
        Console.WriteLine("Min: " + r.Get()[0] + " Max: " + r.Get()[1]);
        // prints "Min: 500 Max: 10"
    }
}

```

## Fixing with an immutable object
Here the example has been modified to prevent changes to the private field by using a `ReadOnlyCollection` object.


```csharp
using System.Collections.ObjectModel;

class Good1
{
    class Range
    {
        private ReadOnlyCollection<int> rarray = new ReadOnlyCollection<int>(new int[2]);

        public Range(int min, int max)
        {
            if (min <= max)
            {
                int[] rarray = new int[2];
                rarray[0] = min;
                rarray[1] = max;
                this.rarray = new ReadOnlyCollection<int>(rarray);
            }
        }

        public ReadOnlyCollection<int> Get() => rarray;
    }
}

```

## Fixing with defensive copying
This is an example of the same class but this time it returns a defensive copy of the private field. There is also a short program showing what happens when an attempt is made to modify the data held by the field.


```csharp
using System;

class Good2
{
    class Range
    {
        private int[] rarray = new int[2];

        public Range(int min, int max)
        {
            if (min <= max)
            {
                rarray[0] = min;
                rarray[1] = max;
            }
        }

        public int[] Get() => (int[])rarray.Clone();
    }

    public static void Main(string[] args)
    {
        Range a = new Range(1, 10);
        int[] a_range = a.Get();
        a_range[0] = 500;
        Console.WriteLine("Min: " + a.Get()[0] + " Max: " + a.Get()[1]);
        // prints "Min: 1 Max: 10"
    }
}

```

## References
* MSDN, C\# Programming Guide, [Arrays as Objects](https://docs.microsoft.com/en-us/dotnet/csharp/programming-guide/arrays/arrays-as-objects).
* MSDN, [ReadOnlyCollection&lt;T&gt;](http://msdn.microsoft.com/en-us/library/ms132474.aspx).
* Common Weakness Enumeration: [CWE-485](https://cwe.mitre.org/data/definitions/485.html).
