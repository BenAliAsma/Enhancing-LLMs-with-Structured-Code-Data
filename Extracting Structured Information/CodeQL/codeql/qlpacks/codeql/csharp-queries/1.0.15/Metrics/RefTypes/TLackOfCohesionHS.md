# Lack of cohesion (HS)
This metric calculates the lack of cohesion of a type using a method proposed by Brian Henderson-Sellers in his book *Object-Oriented Metrics*. Most well designed types will have methods that access the same fields. If methods access disjoint subsets of the class's fields it is an indication the class may have multiple responsibilities.

The Henderson-Sellers method for calculating cohesion works as follows. Let `m` be the number of methods in the type (excluding those that do not access any properties). Let `p ` be the average number of methods that access each field (only for fields that are accessed by methods in the type). LCOM is calculated as follows:

![HS Lack of Cohesion Formula](TLackOfCohesionHSFormula.png)

This gives values between 0 and 2. LCOM values over 0.95 are generally concerning.


## Recommendation
A high lack of cohesion metric often indicates a class has multiple distinct responsibilities. The solution is to identify these responsibilities and to extract the associated methods and data into separate classes.


## Example

```csharp
class TLackOfCohesionCK
{
    private int var1;
    private int var2;
    private int var3;
    private int var4;
    private int var5;

    public void methodA()
    {
        var1 = 1;
        Console.WriteLine(var1);
    }
    public void methodB()
    {
        var2 = 1;
        Console.WriteLine(var2);
    }
    public void methodC()
    {
        var3 = 1;
        Console.WriteLine(var3);
    }
    public void methodD()
    {
        var4 = 1;
        Console.WriteLine(var4);
    }
    public void methodE()
    {
        var4 = 1;
        Console.WriteLine(var4);
        var5 = 1;
        Console.WriteLine(var5);
    }
}

```
This example has an LCOM of 0.95 as shown here:

```
m = 5
p = (1 + 1 + 1 + 1 + 2) / 5 = 1.2
LCOM = (1.2 - 5) / (1 - 5) = 3.8 / 4 = 0.95
```

## References
* Brian Henderson-Sellers. *Object-Oriented Metrics*. Prentice-Hall. 1996.
* Oege de Moor et al. Keynote Address: .QL for Source Code Analysis. Proceedings of the 7th IEEE International Working Conference on Source Code Analysis and Manipulation. 2007.
