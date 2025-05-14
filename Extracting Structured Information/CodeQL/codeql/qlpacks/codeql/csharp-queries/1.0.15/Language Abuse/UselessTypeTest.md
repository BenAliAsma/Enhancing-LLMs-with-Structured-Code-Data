# Useless type test
For any type `B`, an instance of a type derived from `B ` is also an instance of `B`. There is no need to test for this explicitly.


## Recommendation
Remove the useless type test.


## Example
In this example class `Sub` extends class `Super`. A new instance of ` Sub`, `sub` is initialized and checked to see if it is an instance of `Super `. Since `Sub` extends `Super` then there is no need to test that ` sub` is a `Super` because it is guaranteed to be the case.


```csharp
class UselessTypeTest
{
    class Super { }
    class Sub : Super { }

    static void Main(string[] args)
    {
        Sub sub = new Sub();
        if (sub is Super)
        {
            Console.WriteLine("Surprise! sub is a Super!");
        }
    }
}

```
