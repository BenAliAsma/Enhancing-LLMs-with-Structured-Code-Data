# Dubious type test of 'this'
If a type `Derived` inherits (either directly or indirectly) from a type `Base `, then it has a dependency on `Base` by virtue of this inheritance relationship - that is, it cannot be used without `Base` also being present. If, in addition to making `Derived` inherit from `Base`, you also write code that depends on ` Derived` within `Base`, you cause `Base` to depend on `Derived ` as well, resulting in a dependency cycle between the two types. Dependency cycles are a well-known design smell, in that they make code both difficult to read and difficult to test.

In the situation just described, the dependency cycle has been introduced by writing code that checks whether or not `this` is an instance of a derived type. This is a very unwise thing to do - a type should never know about its specific descendants, even though it may of course choose to impose some constraints on them as a group (such as the need for every derived type to implement a method with a specific signature).


## Recommendation
The appropriate solution to this problem is to redesign the base and derived types so that there is no longer a need for the base type to depend on the types that derive from it.


## Example
In this example `BaseClass` attempts to check instances of itself to see which subclass they are and then performs different actions depending on the result. This is very bad practice.


```csharp
class DubiousTypeTestOfThis
{
    class BaseClass
    {
        public int add(int x)
        {
            if (this is FiveAdder)
                return x + 5;

            if (this is TenAdder)
                return x + 10;

            return 0;
        }
    }

    class FiveAdder : BaseClass
    {
        // ...
    }

    class TenAdder : BaseClass
    {
        // ...
    }
}

```
