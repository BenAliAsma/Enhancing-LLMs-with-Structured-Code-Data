# Abstract class only declares common constants
Definitions of constants should be placed in an appropriate class where they belong logically. It is usually bad practice to define an abstract class purely to hold a number of constant definitions.


## Recommendation
This often arises when a developer attempts to put the constant definitions into scope by just extending an abstract class that defines them. While this does save a few characters by not requiring the constant names to be qualified, it is considered bad practice and constants should be referred to by qualified name if needed.


## Example

```csharp
using System;

class Bad
{
    abstract class MathConstants
    {
        public static readonly double Pi = 3.14;
    }

    class Circle : MathConstants
    {
        private double radius;

        public double Area() => Math.Pow(radius, 2) * Pi;
    }
}

```
