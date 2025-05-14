# Cast of 'this' to a type parameter
Casting `this` to a type parameter may indicate an implicit type constraint. That is, the programmer wanted to express that `this` can be converted to the type parameter but could not find an appropriate way to do so. Instead, they rely on derived type to implement the correct interface.


## Recommendation
The solution is to enforce the constraint using the mechanism of an abstract property on the base type. Each derived type must then implement this property, which makes the constraint checkable by the compiler and removes the need for a cast.


## Example
In this example the programmer is relying on any concrete implementations of `BaseNode ` to follow the correct design pattern. `Derived1` does but `Derived2` does not. The program will still compile but will crash if an attempt is made to access the ` Root` property of an instance of `Derived2`.


```csharp
class CastThisToTypeParameter
{
    abstract class BaseNode<T> where T : BaseNode<T>
    {
        public abstract T Parent { get; }

        public T Root
        {
            get
            {
                T cur = (T)this;
                while (cur.Parent != null)
                {
                    cur = cur.Parent;
                }
                return cur;
            }
        }
    }

    class Derived1 : BaseNode<Derived1>
    {
        private string name;
        private Derived1 parent;

        public Derived1(string name, Derived1 parent)
        {
            this.name = name;
            this.parent = parent;
        }

        public override Derived1 Parent { get { return parent; } }
    }

    class Derived2 : BaseNode<Derived1>
    {
        private string name;
        private Derived1 parent;

        public Derived2(string name, Derived1 parent)
        {
            this.name = name;
            this.parent = parent;
        }

        public override Derived1 Parent { get { return parent; } }
    }
}

```
It would be better to enforce this using an abstract property.


```csharp
class CastThisToTypeParameterFix
{
    abstract class BaseNode<T> where T : BaseNode<T>
    {
        public abstract T Self { get; }
        public abstract T Parent { get; }

        public T Root
        {
            get
            {
                T cur = Self;
                while (cur.Parent != null)
                {
                    cur = cur.Parent;
                }
                return cur;
            }
        }
    }

    class ConcreteNode : BaseNode<ConcreteNode>
    {
        private string name;
        private ConcreteNode parent;

        public ConcreteNode(string name, ConcreteNode parent)
        {
            this.name = name;
            this.parent = parent;
        }

        public override ConcreteNode Self { get { return this; } }
        public override ConcreteNode Parent { get { return parent; } }
    }
}

```
