# Cyclomatic complexity of functions
This metric measures the number of linearly independent execution paths through methods. Linearly independent paths are calculated using the control flow diagram for a piece of code. A linearly independent path includes at least one new edge that has not been included in any linearly independent path counted so far. Methods with a high cyclomatic complexity can be difficult to understand and difficult to test because there are so many different ways they could execute.

Consider this method:


```csharp
public static void foo(int count)
{
    if (count > 10) {
        Console.WriteLine("The count is large");
    }

    var timesLeft = count;
    while (timesLeft > 0) {
        switch(Console.ReadLine()) {
            case "BYE" : Console.WriteLine("Good bye"); break;
            case "HELLO" : Console.WriteLine("Hi"); break;
            case "HELP" : Console.WriteLine("Try HELLO or BYE."); break;
            default : Console.WriteLine("Input not understood."); break;
        }
        timesLeft--;
    }
}

```
The control flow diagram for through this method looks like this:

![Control Flow Diagram](CCyclomaticComplexity.png)The first linearly independent path through this method is one where the condition for every branch point returns false. The path through the diagram would go Start &rarr; count&gt;10 &rarr; timesLeft&gt;0 &rarr; End. There is also another path where the first condition returns true and hence "The count is large" is printed. This counts as a second independent path because it adds the edges from count&gt;10 &rarr; "The count is large" &rarr; timesLeft&gt;0 which have not been included in the first path we counted. Likewise the second condition being true would add another independent path through the switch statement and one of its cases. The switch statement has 4 cases that count as paths however one has already been counted in the previous path. As such the switch statement adds another 3 independent paths through the method bringing the total cyclomatic complexity to 6.


## Recommendation
Complex methods should have parts of their functionality extracted to helper methods. This makes testing easier because each helper method can be tested individually.


## References
* Wikipedia. [Cyclomatic complexity](http://en.wikipedia.org/wiki/Cyclomatic_complexity).
* Wikipedia. [Control flow diagram](http://en.wikipedia.org/wiki/Control_flow_diagram).
* Wolfram MathWorld. [Linearly Independent](http://mathworld.wolfram.com/LinearlyIndependent.html).
