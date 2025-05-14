# Container contents are never accessed
If the contents of a collection are never used, then it is useless and therefore unnecessary. This adds performance overhead, obscures the code, and may indicate an error in the logic.


## Recommendation
Either remove the collection if it is no longer needed, or ensure that it is used as intended.


## Example
In this example, the property `Names` returns the wrong collection (`genres`). This logic error means that the `names` collection is populated but never accessed.


```csharp
class Composers
{
    IList<string> names, genres;

    public Composers()
    {
        names = new List<string> { "Bach", "Beethoven", "Chopin" };
        genres = new List<string> { "Classical", "Romantic", "Jazz" };
    }

    public IList<string> Names
    {
        get { return genres; }
    }

    public IList<string> Genres
    {
        get { return genres; }
    }
}

```
The code is fixed by returning the correct field for `Names`.


```csharp
class Composers
{
    IList<string> names, genres;

    public Composers()
    {
        names = new List<string> { "Bach", "Beethoven", "Chopin" };
        genres = new List<string> { "Classical", "Romantic", "Jazz" };
    }

    public IList<string> Names
    {
        get { return names; }
    }

    public IList<string> Genres
    {
        get { return genres; }
    }
}

```
