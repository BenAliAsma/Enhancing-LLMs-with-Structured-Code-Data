# Virtual call in constructor or destructor
Virtual calls or accesses in a constructor or destructor might not behave as expected. The constructor for the base class is always executed first, but in the context of the runtime type of the object. If a method is overridden in a sub type then that overridden method can be called from the constructor of the base type. This can lead to the methods of a class being called before the constructor of the class, which can have unexpected consequences.


## Recommendation
Carefully check the virtual calls or accesses to make sure they will behave as expected. If possible, eliminate the virtual calls.


## Example
In this example `DClass.Method()` is called before the `DClass` constructor is called. This is a problem because `DClass`'s version of `Method()` is not expecting to be called before the constructor.


```csharp
class VirtualCallInConstructorOrDestructor
{
    class BaseClass
    {
        protected String classReady = "No";
        public BaseClass()
        {
            Console.WriteLine("Base constructor called.");
            Method();
        }

        public virtual void Method()
        {
            Console.WriteLine("Base method called.");
        }
    }
    class DClass : BaseClass
    {
        public DClass()
        {
            Console.WriteLine("D constructor called.");
            classReady = "Yes";
        }

        public override void Method()
        {
            Console.WriteLine("D method called. Ready for method to be called? " + classReady);
        }
    }
    public static void Main(string[] args)
    {
        BaseClass x = new DClass();
    }
}

```
This example outputs the following:

```
Base constructor called.
D method called. Ready for method to be called? No
D constructor called.
```
