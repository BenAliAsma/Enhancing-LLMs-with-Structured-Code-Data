# Feature envy
*Feature envy* refers to situations where a method is "in the wrong place", because it does not use many methods or variables of its own class, but uses a whole range of methods or variables from some other class. This violates the principle of putting data and behavior in the same place, and exposes internals of the other class to the method.


## Recommendation
For each method that may exhibit feature envy, see if it needs to be declared in its present location, or if you can move it to the class it is "envious" of. A common example is a method that calls a large number of getters on another class to perform a calculation that does not rely on anything from its own class. In such cases, you should move the method to the class containing the data. If the calculation depends on some values from the method's current class, they can either be passed as arguments or accessed using getters from the other class.

If it is inappropriate to move the entire method, see if all the dependencies on the other class are concentrated in just one part of the method. If so, you can move them into a method of their own. You can then move this method to the other class and call it from the original method.

If a class is envious of functionality defined in a superclass, perhaps the superclass needs to be rewritten to become more extensible and allow its subtypes to define new behavior without them depending so deeply on the superclass's implementation. The *template method* pattern may be useful in achieving this.

Modern IDEs provide several refactorings that may be useful in addressing instances of feature envy, typically under the names of "Move method" and "Extract method".

Occasionally, behavior can be misinterpreted as feature envy when in fact it is justified. The most common examples are complex design patterns like *visitor* or *strategy*, where the goal is to separate data from behavior.


## Example
In the following example, the method `GetTotalPrice` is in the `Basket` class, but it only uses data belonging to the `Item` class. Therefore, it represents an instance of feature envy.


```csharp
using System;

class Bad
{
    class Item
    {
        public bool IsOutOfStock;
        public decimal Price;
        public decimal Tax;
        public bool IsOnSale;
        public decimal SaleDiscount;
    }

    class Basket
    {
        decimal GetTotalPrice(Item i)
        {
            if (i.IsOutOfStock)
                throw new Exception("Item ${i} is out of stock.");
            var price = i.Price + i.Tax;
            if (i.IsOnSale)
                price = price - i.SaleDiscount * price;
            return price;
        }
    }
}

```
In the revised example, `GetTotalPrice` is moved to the `Item` class and its parameter is removed.


```csharp
using System;

class Good
{
    class Item
    {
        public bool IsOutOfStock;
        public decimal Price;
        public decimal Tax;
        public bool IsOnSale;
        public decimal SaleDiscount;

        decimal GetTotalPrice(Item i)
        {
            if (i.IsOutOfStock)
                throw new Exception("Item ${i} is out of stock.");
            var price = i.Price + i.Tax;
            if (i.IsOnSale)
                price = price - i.SaleDiscount * price;
            return price;
        }
    }
}

```

## References
* E. Gamma, R. Helm, R. Johnson, J. Vlissides, *Design patterns: elements of reusable object-oriented software*. Addison-Wesley Longman Publishing Co., Inc., Boston, MA, 1995.
* W. C. Wake, *Refactoring Workbook*, pp. 93&ndash;94. Addison-Wesley Professional, 2004.
