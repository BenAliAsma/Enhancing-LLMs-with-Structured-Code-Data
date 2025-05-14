# Unchecked return value
Ignoring a method's return value can be a security risk and a common source of defects. If an attacker can force the method to fail, the subsequent program logic could lead to a vulnerability because the program will be running in an unexpected state. This rule checks that the return value of standard library methods like `System.IO.Stream.Read(...)` is always used. Furthermore, it identifies any calls to methods that ignore the return value despite it being frequently used elsewhere. That is, if more than 90% of the total calls to a particular method use its return value, all other calls that discard its return value are flagged as suspicious.


## Recommendation
Check the return value of the method and take appropriate action.


## Example
In the method `IgnoreOne`, there are lots of calls to `DoPrint` where the return value is checked. However, the return value of `DoPrint("J")` is not checked.

In the method `IgnoreRead`, the result of the call to `FileStream.Read` is ignored. This is dangerous, as the amount of data read might be less than the length of the array it is being read into.


```csharp
using System;
using System.IO;

class Bad
{
    public bool DoPrint(string s) => true;

    public void IgnoreOne()
    {
        if (DoPrint("A"))
            Console.WriteLine("A");
        if (DoPrint("B"))
            Console.WriteLine("B");
        if (DoPrint("C"))
            Console.WriteLine("C");
        if (DoPrint("D"))
            Console.WriteLine("D");
        if (DoPrint("E"))
            Console.WriteLine("E");
        if (DoPrint("F"))
            Console.WriteLine("F");
        if (DoPrint("G"))
            Console.WriteLine("G");
        if (DoPrint("H"))
            Console.WriteLine("H");
        if (DoPrint("I"))
            Console.WriteLine("I");

        DoPrint("J");
    }

    void IgnoreRead(string path)
    {
        var file = new byte[10];
        using (var f = new FileStream(path, FileMode.Open))
            f.Read(file, 0, file.Length);
    }
}

```

## References
* Common Weakness Enumeration: [CWE-252](https://cwe.mitre.org/data/definitions/252.html).
