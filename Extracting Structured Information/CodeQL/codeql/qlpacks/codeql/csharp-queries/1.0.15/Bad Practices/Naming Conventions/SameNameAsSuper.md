# Class has same name as super class
This rule identifies a class that has the same name as a class it extends. This is likely to cause confusion.


## Recommendation
Change the name of the subclass to make it more clear what its purpose is. It is normally possible to express how it is different to a superclass.


## Example
In this example references to ArrayList later on in the Players class could be confusing.


```csharp
class Players
{
    private ArrayList playerList;
    class ArrayList : System.Collections.ArrayList
    {
        public String GetBestThree()
        {
            return "1st: " + this[0] + "\n 2nd:" + this[1] + "\n 3rd:" + this[2];
        }
    }
}

```

## Fixing by Using a More Descriptive Name
The example could be easily fixed by changing the `Players.ArrayList` class name to `Players.PlayerList`. This is more descriptive and it avoids confusion with the `ArrayList` class from `Collections`.


## References
* Robert C. Martin, *Clean Code - A handbook of agile software craftsmanship*, &sect;17.N4. Prentice Hall, 2008.
