# Nesting depth
This metric measures the maximum nesting depth of statements. When statements are nested too deeply in the code they become difficult to understand. This is because understanding a line that is deeply nested requires an understanding of the context of that line.


## Recommendation
The best way to address this problem is to extract more deeply nested parts of the code into their own logical methods.


## Example
This method has a nesting depth of 4.


```csharp
public void printCharacterCodes(string[] strings)
{
    if (strings != null)
    {
        foreach(string s in strings){
            if (s != null)
            {
                foreach (char c in s)
                {
                    Console.WriteLine(c + "=" + (int)c);
                }
            }
        }
    }
}

```

## Fixed by Extraction
This code is easier to read because the part that prints every character in the string along with its character code has been extracted to a separate method.


```csharp
public static void PrintAllCharInts(string s){
    if (s != null)
    {
        foreach (char c in s)
        {
            Console.WriteLine(c + "=" + (int)c);
        }
    }
}
public static void Main(string[] args)
{
    string[] strings = new string[5];
    if (strings != null)
    {
        foreach(string s in strings){
            PrintAllCharInts(s);
        }
    }
}

```

## References
* Martin Fowler. *Refactoring*. pp. 89-95. Addison-Wesley. 1999.
* Steve McConnell - *Code Complete: A Practical Handbook of Software Construction*.
