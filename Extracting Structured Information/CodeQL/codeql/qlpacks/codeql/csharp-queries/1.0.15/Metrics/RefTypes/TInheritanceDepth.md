# Inheritance depth
This metric measures the inheritance depth of reference types. Inheritance is a great way to reduce code duplication but when a reference type is nested too deeply in an inheritance hierarchy the code can become difficult to manage. Firstly the code is more difficult to understand because the superclasses have to be understood first to fully understand the reference types at the bottom of the hierarchy. Secondly the code becomes very difficult to update; making a change at the top of the hierarchy could break a lot of the types lower down.


## Recommendation
A high inheritance depth is an indicator that the inheritance hierarchy might be badly structured. Reviewing the hierarchy might show a number of issues.

One possible issue is that a class represents the same abstraction as its superclass. In this case the two classes should simply be merged into one.

It is also possible that the class hierarchy is actually trying to represent multiple properties of the same object. If this is the case then it may be preferable to use composition instead of inheritance.


## Example
In this example the inheritance hierarchy is being used to represent multiple properties of a `Book`. Inheritance should not be used in this way.


```csharp
class Book
{
    public void read()
    {
        // ...
    }
}
abstract class BlueBook : Book { /* ... */ }
abstract class RedBook : Book { /* ... */ }
abstract class LongRedBook : RedBook { /* ... */ }
abstract class ShortRedBook : RedBook { /* ... */ }
abstract class LongBlueBook : BlueBook { /* ... */ }
abstract class ShortBlueBook : BlueBook { /* ... */ }
class ShortBlueNovel : ShortBlueBook { /* ... */ }
// ...

```

## Fixing by Using Composition
This code is more appropriate because different properties of `Book`s are defined by fields rather than the position in the inheritance hierarchy.


```csharp
class Color{ /* ... */ }
class BookLength{ /* ... */ }
class BookType{ /* ... */ }
class Book
{
    private Color color;
    private BookLength length;
    private BookType type;
    public Book(Color color, BookLength length, BookType type){
        this.color = color;
        this.length = length;
        this.type = type;
    }
    public void read()
    {
        // ...
    }
}

```

## References
* Barbara Liskov. [Data Abstraction and Hierarchy](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.12.819&rep=rep1&type=pdf).
* Martin Fowler. *Refactoring*. Addison-Wesley. 1999.
* Mick West. [Evolve Your Hierarchy](http://cowboyprogramming.com/2007/01/05/evolve-your-heirachy).
