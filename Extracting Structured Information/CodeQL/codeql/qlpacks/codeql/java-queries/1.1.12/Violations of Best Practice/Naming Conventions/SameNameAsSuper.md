# Class has same name as super class
A class that has the same name as its superclass may be confusing.


## Recommendation
Clarify the difference between the subclass and the superclass by using different names.


## Example
In the following example, it is not clear that the `attendees` field refers to the inner class `Attendees` and not the class `com.company.util.Attendees`.


```java
import com.company.util.Attendees;

public class Meeting
{
	private Attendees attendees;

	// ...
	// Many lines
	// ...

	// AVOID: This class has the same name as its superclass.
	private static class Attendees extends com.company.util.Attendees
	{
		// ...
	}
}
```
To fix this, the inner class should be renamed.


## References
* R. C. Martin, *Clean Code: A Handbook of Agile Software Craftsmanship*, &sect;17.N4. Prentice Hall, 2008.
