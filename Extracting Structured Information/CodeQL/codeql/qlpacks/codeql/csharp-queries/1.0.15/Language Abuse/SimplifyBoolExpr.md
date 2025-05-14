# Unnecessarily complex Boolean expression
There are a number of Boolean expression patterns that can easily be rewritten to make them simpler. Boolean expressions involving comparisons with Boolean literals, ternary conditionals with a Boolean literal as one of the results, double negations, or negated comparisons can all be changed to equivalent and simpler expressions.


## Recommendation
If `A` and `B` are expressions of Boolean type, you can simplify them using the rewrites shown below.

<table><tbody> <tr><th>Expression</th><th></th><th>Simplified expression</th></tr> <tr><td><code>A == true</code></td><td></td><td><code>A</code></td></tr> <tr><td><code>A != false</code></td><td></td><td><code>A</code></td></tr> <tr><td><code>A == false</code></td><td></td><td><code>!A</code></td></tr> <tr><td><code>A != true</code></td><td></td><td><code>!A</code></td></tr> <tr><td><code>A ? true : B</code></td><td></td><td><code>A || B</code></td></tr> <tr><td><code>A ? B : false</code></td><td></td><td><code>A &amp;&amp; B</code></td></tr> <tr><td><code>A ? B : true</code></td><td></td><td><code>!A || B</code></td></tr> <tr><td><code>A ? false : B</code></td><td></td><td><code>!A &amp;&amp; B</code></td></tr> <tr><td><code>A ? true : false</code></td><td></td><td><code>A</code></td></tr> <tr><td><code>A ? false : true</code></td><td></td><td><code>!A</code></td></tr> <tr><td><code>!!A</code></td><td></td><td><code>A</code></td></tr> <tr><td><code>A &amp;&amp; true</code></td><td></td><td><code>A</code></td></tr> <tr><td><code>A || false</code></td><td></td><td><code>A</code></td></tr> </tbody></table>
Some expressions always yield a constant value. If the side-effect in `A` is intended, consider restructuring the code to make this more clear. Otherwise, replace the expression with the constant value as shown below.

<table><tbody> <tr><th>Expression</th><th></th><th>Value</th></tr> <tr><td><code>A &amp;&amp; false</code></td><td></td><td><code>false</code></td></tr> <tr><td><code>A || true</code></td><td></td><td><code>true</code></td></tr> <tr><td><code>A ? true : true</code></td><td></td><td><code>true</code></td></tr> <tr><td><code>A ? false : false</code></td><td></td><td><code>false</code></td></tr> </tbody></table>
In addition to the rewrites above, negated comparisons can also be simplified in the following way:

<table><tbody> <tr><th>Expression</th><th></th><th>Simplified expression</th></tr> <tr><td><code>!(A == B)</code></td><td></td><td><code>A != B</code></td></tr> <tr><td><code>!(A != B)</code></td><td></td><td><code>A == B</code></td></tr> <tr><td><code>!(A &lt; B)</code></td><td></td><td><code>A &gt;= B</code></td></tr> <tr><td><code>!(A &gt; B)</code></td><td></td><td><code>A &lt;= B</code></td></tr> <tr><td><code>!(A &lt;= B)</code></td><td></td><td><code>A &gt; B</code></td></tr> <tr><td><code>!(A &gt;= B)</code></td><td></td><td><code>A &lt; B</code></td></tr> </tbody></table>

## Example
In the following example, the properties `Espresso`, `Latte`, and `Grande` are written in a complex way and can be simplified.


```csharp
class Bad
{
    int Size { get; set; }

    bool Espresso => !(Size > 4);
    bool Latte => Espresso == false && Size <= 8;
    bool Grande => Espresso == false ? Latte != true : false;
}

```
The code below shows the same logic expressed in a simpler and more readable way.


```csharp
class Good
{
    int Size { get; set; }

    bool Espresso => Size <= 4;
    bool Latte => !Espresso && Size <= 8;
    bool Grande => !Espresso && !Latte;
}

```

## References
* Microsoft C\# Reference: [! Operator](https://docs.microsoft.com/en-us/dotnet/csharp/language-reference/operators/logical-negation-operator), [== Operator](https://docs.microsoft.com/en-us/dotnet/csharp/language-reference/operators/equality-comparison-operator), [!= Operator](https://docs.microsoft.com/en-us/dotnet/csharp/language-reference/operators/not-equal-operator), [&amp;&amp; Operator](https://docs.microsoft.com/en-us/dotnet/csharp/language-reference/operators/conditional-and-operator), [|| Operator](https://docs.microsoft.com/en-us/dotnet/csharp/language-reference/operators/conditional-or-operator), [?: Operator](https://docs.microsoft.com/en-us/dotnet/csharp/language-reference/operators/conditional-operator), [&lt; Operator](https://docs.microsoft.com/en-us/dotnet/csharp/language-reference/operators/less-than-operator).
