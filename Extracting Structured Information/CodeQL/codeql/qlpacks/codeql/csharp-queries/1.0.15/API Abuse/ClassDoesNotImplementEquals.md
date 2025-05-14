# Class does not implement Equals(object)
When the class of the object on which `Equals(object)` is called does not define its own `Equals(object)` method, an `Equals(object)` method defined in one of its base classes will be called instead. In the worst case, the `Equals(object)` method of `System.Object` will be called, resulting in a reference equality check. This is probably not what was intended.

Classes that implement the `==` operator should also override the `Equals(object)` method, because otherwise the two forms of equality will behave differently, leading to unexpected behavior.


## Recommendation
Implement an `Equals(object)` method for the highlighted class. Examine subclasses of the class highlighted to determine if they should implement their own equals method too.


## Example
The output of this example states that "car1 does equal car2" despite the fact that one is a leaded version and one is an unleaded version. This is because the `GasolineCar` class is inheriting `Equals(object)` from `Car` and that method states that two `Car`s are equal if their make and model are the same.


```csharp
using System;

class Bad
{
    class Car
    {
        protected string make;
        protected string model;

        public Car(string make, string model)
        {
            this.make = make;
            this.model = model;
        }

        public override bool Equals(object obj)
        {
            if (obj is Car c && c.GetType() == typeof(Car))
                return make == c.make && model == c.model;
            return false;
        }
    }

    class GasolineCar : Car
    {
        protected bool unleaded;

        public GasolineCar(string make, string model, bool unleaded) : base(make, model)
        {
            this.unleaded = unleaded;
        }
    }

    public static void Main(string[] args)
    {
        var car1 = new GasolineCar("Ford", "Focus", true);
        var car2 = new GasolineCar("Ford", "Focus", false);
        Console.WriteLine("car1 " + (car1.Equals(car2) ? "does" : "does not") + " equal car2.");
    }
}

```
In the revised example, `GasolineCar` overrides `Equals(object)`, and the output is "car1 does not equal car2", as expected.


```csharp
using System;

class Good
{
    class Car
    {
        protected string make;
        protected string model;

        public Car(string make, string model)
        {
            this.make = make;
            this.model = model;
        }

        public override bool Equals(object obj)
        {
            if (obj is Car c && c.GetType() == typeof(Car))
                return make == c.make && model == c.model;
            return false;
        }
    }

    class GasolineCar : Car
    {
        protected bool unleaded;

        public GasolineCar(string make, string model, bool unleaded) : base(make, model)
        {
            this.unleaded = unleaded;
        }

        public override bool Equals(object obj)
        {
            if (obj is GasolineCar gc && gc.GetType() == typeof(GasolineCar))
                return make == gc.make && model == gc.model && unleaded == gc.unleaded;
            return false;
        }
    }

    public static void Main(string[] args)
    {
        var car1 = new GasolineCar("Ford", "Focus", true);
        var car2 = new GasolineCar("Ford", "Focus", false);
        Console.WriteLine("car1 " + (car1.Equals(car2) ? "does" : "does not") + " equal car2.");
    }
}

```

## References
* Microsoft: [Equality Operators](https://docs.microsoft.com/en-us/dotnet/standard/design-guidelines/equality-operators), [Equality Comparisons (C\# Programming Guide)](https://docs.microsoft.com/en-us/dotnet/csharp/programming-guide/statements-expressions-operators/equality-comparisons).
