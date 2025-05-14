# Variable name is too short
Source code with lots of short variable names is likely to be difficult to read.


## Recommendation
There are some situations where short variable names may be acceptable. If the code is calculating a mathematical formula with commonly accepted letters representing variables then it may be acceptable to use single letter variable names because it makes it easier to compare the program with the original formula. It is important to consider how long a variable will be used for when deciding how descriptive its name needs to be. If the variable is declared on one line and only used on the next few then it may be clear what its purpose is even if it has a short variable name. If a variable does not fit in one of these exception categories then consider changing the names of the variables to make it more clear what their purpose is.


## Example
This example demonstrates the readability issues of short variables.


```csharp
class VariableNameTooShort
{
    public static void Main(string[] args)
    {
        int q = 5;
        int s = 2;
        int x = 15;
        int qx = q * x;
        // later on in the code
        int r = q + (2 * qx) / (x + s);
    }
}

```
Here is an example of a program where using short variable names may be acceptable. This is because the program is using a formula with commonly accepted letter substitutions and the life of the variables is very short. It also includes a comment to explain the variables to someone who might not know what they mean.


```csharp
class VariableNameTooShortAcceptableCase
{
    public static void Main(string[] args)
    {
        /*
         * u - object starting position
         * t - time
         * a - acceleration
         * s - speed
         */
        double u = 5;
        double t = 2;
        double a = 9.81;
        double s = (u * t) + ((1.0 / 2.0) * a * Math.Pow(t, 2));
    }
}

```

## References
* Ars Technica. [Is there an excuse for short variable names?](http://arstechnica.com/information-technology/2013/02/is-there-an-excuse-for-short-variable-names/).
