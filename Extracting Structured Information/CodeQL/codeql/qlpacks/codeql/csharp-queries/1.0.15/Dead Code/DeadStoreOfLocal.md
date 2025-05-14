# Useless assignment to local variable
A value is assigned to a local variable, but either that variable is never read later on, or its value is always overwritten before being read. This means that the original assignment has no effect, and could indicate a logic error or incomplete code.


## Recommendation
Ensure that you check the program logic carefully. If a value is really not needed, consider omitting the assignment. Be careful, though: if the right-hand side has a side effect (like performing a method call), it is important to keep this to preserve the overall behavior.


## Example
The following example shows six different types of assignments to local variables whose value is not read:

* In `ParseInt`, the result of the call to `int.TryParse` is assigned directly to the unread local variable `success`.
* In `IsDouble`, the `out` argument of the call to `int.TryParse` is assigned to the unread local variable `i`.
* In `ParseDouble`, the exception thrown by the call to `double.Parse` in case of parse failure is assigned to the unread local variable `e`.
* In `Count`, the elements of `ss` are assigned to the unread local `foreach` variable `s`.
* In `IsInt`, `o` is assigned (in case `o` is an integer) to the unread local type test variable `i`.
* In `IsString`, `o` is assigned (in case `o` is a string) to the unread local type case variable `s`.

```csharp
using System;

class Bad
{
    double ParseInt(string s)
    {
        var success = int.TryParse(s, out int i);
        return i;
    }

    bool IsDouble(string s)
    {
        var success = double.TryParse(s, out double i);
        return success;
    }

    double ParseDouble(string s)
    {
        try
        {
            return double.Parse(s);
        }
        catch (FormatException e)
        {
            return double.NaN;
        }
    }

    int Count(string[] ss)
    {
        int count = 0;
        foreach (var s in ss)
            count++;
        return count;
    }

    string IsInt(object o)
    {
        if (o is int i)
            return "yes";
        else
            return "no";
    }

    string IsString(object o)
    {
        switch (o)
        {
            case string s:
                return "yes";
            default:
                return "no";
        }
    }
}

```
The revised example eliminates the unread assignments.


```csharp
using System;

class Good
{
    double ParseInt(string s)
    {
        int.TryParse(s, out int i);
        return i;
    }

    bool IsDouble(string s)
    {
        var success = double.TryParse(s, out _);
        return success;
    }

    double ParseDouble(string s)
    {
        try
        {
            return double.Parse(s);
        }
        catch (FormatException)
        {
            return double.NaN;
        }
    }

    int Count(string[] ss)
    {
        return ss.Length;
    }

    string IsInt(object o)
    {
        if (o is int)
            return "yes";
        else
            return "no";
    }

    string IsString(object o)
    {
        switch (o)
        {
            case string _:
                return "yes";
            default:
                return "no";
        }
    }
}

```

## References
* Wikipedia: [Dead store](http://en.wikipedia.org/wiki/Dead_store).
* MSDN, Code Analysis for Managed Code, [CA1804: Remove unused locals](http://msdn.microsoft.com/en-us/library/ms182278.aspx).
* Microsoft: [What's new in C\# 7 - Discards](https://docs.microsoft.com/en-us/dotnet/csharp/whats-new/csharp-7#discards).
* Common Weakness Enumeration: [CWE-563](https://cwe.mitre.org/data/definitions/563.html).
