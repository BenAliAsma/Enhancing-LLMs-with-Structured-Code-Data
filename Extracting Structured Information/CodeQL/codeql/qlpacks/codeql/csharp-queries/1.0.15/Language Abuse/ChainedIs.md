# Chain of 'is' tests
Long sequences of type tests are primarily used to dispatch to different bits of code based on the type of a variable, as shown in the example later in this topic. They are often used to simulate pattern-matching in languages that do not support it. Whilst they do work as a dispatch method, they have a number of disadvantages:

* They are difficult to maintain, because it is easy to add a new subtype and forget to modify all of the type test sequences throughout your code.
* They introduce unwanted dependencies on concrete classes. Code that could be written only in terms of an interface must now take account of all the special cases.
* They can be error-prone - if you inadvertently test for a base type before a derived type, the code that handles the derived type is never executed.
* They can exhibit poor performance, because they effectively iterate through a list of types in linear time, checking each one in turn.

## Recommendation
There are a number of possible solutions to the problem, depending on the exact circumstances:

* **Polymorphism**. This involves adding a virtual method to the type hierarchy and putting the bits of code to be called in the relevant override for each concrete class. It is a tidy solution when you can change the type hierarchy in question and the operation being implemented is a core part of the functionality that the types in question should implement. It is important to be careful not to introduce unwanted dependencies if you choose this approach - if the operation depends on things that themselves depend on the type hierarchy, then you cannot move the operation to the type hierarchy without creating a dependency cycle.
* **The visitor pattern**. This involves introducing a visitor interface containing a `visit` method for each type in the type hierarchy, and adding an `accept` method to each type in the hierarchy that takes such a visitor as its parameter. Each type's `accept` method calls the visitor's `visit` method on `this`. Concrete visitors then implement the interface and do whatever is necessary for each specific type. This is a good solution when you can change the type hierarchy in question and the type hierarchy should not know about the operation being implemented, either for dependency reasons or because it is not part of the core functionality of the types in the hierarchy. It is also sensible if you want to provide multiple operations with the same structure on the same set of types and you want the types themselves to control the way in which the operation is structured (for example 'visit this tree using an in-order walk and do whatever is necessary for each node'). The disadvantages are that the basic visitor pattern is cyclically-dependent, and that the infrastructure involved is comparatively heavyweight.
* **Reflection**. This involves looking up one of a set of overloaded methods based on the type of one of the method parameters, and calling it manually. This option should never be the first choice because it necessitates a loss of type safety and is rather untidy, but there are times when it can be sensible. In particular, it is useful when you cannot change the type hierarchy in question (for example because it is third-party code) and your code must compile with versions of C\# that do not support `dynamic` (pre-4.0).
* **Use `dynamic`**. This involves converting (either explicitly or implicitly) the type of the object to `dynamic` so as to perform a dynamically-resolved call on it. It is only an option in C\# 4.0 and later. As with reflection, it necessitates a loss of type safety, although it is somewhat cleaner from a syntactic perspective. It is a useful approach when you cannot change the type hierarchy in question or you are keen to avoid a heavyweight solution like the visitor pattern (you are effectively losing some type safety to gain some readability).

## Example
This example uses a series of chained `is` statements to perform a different action depending on what kind of `Animal` is being iterated over.


```csharp
class ChainedIs
{
    interface Animal { }
    class Cat : Animal { }
    class Dog : Animal { }

    public static void Main(string[] args)
    {
        List<Animal> animals = new List<Animal> { new Cat(), new Dog() };
        foreach (Animal a in animals)
        {
            if (a is Cat)
                Console.WriteLine("Miaow!");
            else if (a is Dog)
                Console.WriteLine("Woof!");
            else
                throw new Exception("Oops!");
        }
    }
}

```
Polymorphism is illustrated in the example below.


```csharp
class ChainedIs
{
    interface Animal
    {
        void Speak();
    }
    class Cat : Animal
    {
        public void Speak() { Console.WriteLine("Miaow!"); }
    }
    class Dog : Animal
    {
        public void Speak() { Console.WriteLine("Woof!"); }
    }

    public static void Main(string[] args)
    {
        List<Animal> animals = new List<Animal> { new Cat(), new Dog() };
        foreach (var a in animals)
            a.Speak();
    }
}

```
Here is the same example again using the visitor pattern. This is a better solution if the idea of animal noises should be separate from the idea of animals.


```csharp
class ChainedIs
{
    interface Visitor
    {
        void Visit(Cat c);
        void Visit(Dog d);
    }
    class SpeakVisitor : Visitor
    {
        public void Visit(Cat c) { Console.WriteLine("Miaow!"); }
        public void Visit(Dog d) { Console.WriteLine("Woof!"); }
    }
    interface Animal
    {
        void Accept(Visitor v);
    }
    class Cat : Animal
    {
        public void Accept(Visitor v) { v.Visit(this); }
    }
    class Dog : Animal
    {
        public void Accept(Visitor v) { v.Visit(this); }
    }

    public static void Main(string[] args)
    {
        List<Animal> animals = new List<Animal> { new Cat(), new Dog() };
        foreach (var a in animals)
            a.Accept(new SpeakVisitor());
    }
}

```
More details on reflection and the use of `dynamic` can be found in the references.


## References
* J. Albahari and B. Albahari, *C\# 4.0 in a Nutshell - The Definitive Reference*, Chapters 18 and 19, O'Reilly Media, 2010.
* R. Johnson, J. Vlissides, R. Helm and E. Gamma, *Design Patterns: Elements of Reusable Object-Oriented Software*, Addison-Wesley Professional, 1994.
