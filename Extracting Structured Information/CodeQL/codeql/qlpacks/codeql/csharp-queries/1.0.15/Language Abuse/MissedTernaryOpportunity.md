# Missed ternary opportunity
An `if` statement where both branches do nothing but return or write to a variable can be difficult to read. It takes up a lot of lines of code and, in the case of assignment, requires the declaration of the variable to be on a different line to the two assignments. It also does not adequately express the intent of the programmer to assign or return a value based on a condition.


## Recommendation
This pattern can be better expressed using the ternary (`?`) operator. This solves all the above problems by making shorter code that is easier to read and better expresses the intent of the programmer.


## Example
In this example the `if` statement controls only what is returned by the method.


```csharp
class MissedTernaryOpportunity
{
    static int MyAbs(int x)
    {
        if (x >= 0)
            return x;
        else
            return -x;
    }
}

```
It could be expressed a lot more simply using the ternary operator.


```csharp
class MissedTernaryOpportunityFix
{
    static int MyAbs(int x)
    {
        return x >= 0 ? x : -x;
    }
}

```
