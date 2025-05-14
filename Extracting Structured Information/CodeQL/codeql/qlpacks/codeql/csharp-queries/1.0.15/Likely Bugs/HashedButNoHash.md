# Hashed value without GetHashCode definition
Classes that override `System.Object.Equals()` but not `System.Object.GetHashCode()` can yield unexpected results if instances of those classes are stored in a hashing data structure.


## Recommendation
Override the `GetHashCode` method such that for two instances a and b, where a.Equals(b) is true, a.GetHashCode() and b.GetHashCode() are equal. The C\# documentation states \[1\]:

> If two objects compare as equal, the GetHashCode method for each object must return the same value. However, if two objects do not compare as equal, the GetHashCode methods for the two objects do not have to return different values.


## Example

```csharp
using System;
using System.Collections;

class Point
{
    private int x;
    private int y;

    public Point(int x, int y)
    {
        this.x = x;
        this.y = y;
    }

    public override bool Equals(Object other)
    {
        Point point = other as Point;
        if (point == null)
        {
            return false;
        }
        return this.x == point.x && this.y == point.y;
    }

    public static void Main(string[] args)
    {
        Hashtable hashtable = new Hashtable();
        hashtable[new Point(5, 4)] = "A point"; // BAD
        // Point overrides the Equals method but not GetHashCode.
        // As such it is probably not useful to use one as the key for a Hashtable.
        Console.WriteLine(hashtable[new Point(5, 4)]);
    }
}

```

## References
* MSDN, C\# Programmer's Reference, [Object.GetHashCode](http://msdn.microsoft.com/en-us/library/system.object.gethashcode.aspx)
