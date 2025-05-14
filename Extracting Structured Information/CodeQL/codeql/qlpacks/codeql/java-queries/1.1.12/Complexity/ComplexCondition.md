# Complex condition
In general, very complex conditions are difficult to write and read, and increase the chance of defects.


## Recommendation
Firstly, a condition can often be simplified by changing other parts of the code to initialize variables more consistently. For example, is there a semantic difference between `id` being `null` and having zero-length? If not, choosing one sentinel value and using it consistently simplifies most uses of that variable.

Secondly, extracting part of a condition into a Boolean-valued method can simplify the condition and also allow code reuse, with all its benefits.

Thirdly, assigning each subcondition of the condition to a local variable, and then using the variables in the condition instead can simplify the condition.


## Example
The following example shows a complex condition found in a real program used by millions of people. The condition is so confusing that even the programmer who wrote it is not sure if he got it right (see the `TODO` comment).


```java
public class Dialog
{
	// ...

	private void validate() {
		// TODO: check that this covers all cases
		if ((id != null && id.length() == 0) ||
			((partner == null || partner.id == -1) &&
			((option == Options.SHORT && parameter.length() == 0) ||
			(option == Options.LONG && parameter.length() < 8))))
		{
			disableOKButton();
		} else {
			enableOKButton();
		}
	}

	// ...
}
```
The condition can be simplified by extracting parts of the condition into Boolean-valued methods. These methods are then used in the condition.


```java
public class Dialog
{
    // ...

    private void validate() {
      if(idIsEmpty() || (noPartnerId() && parameterLengthInvalid())){ // GOOD: Condition is simpler
        disableOKButton();
      } else {
        enableOKButton();
      }
    }

    private boolean idIsEmpty(){
      return id != null && id.length() == 0;
    }

    private boolean noPartnerId(){
      return partner == null || partner.id == -1;
    }

    private boolean parameterLengthInvalid(){
      return (option == Options.SHORT && parameter.length() == 0) ||
             (option == Options.LONG && parameter.length() < 8);
    }

    // ...
}    
```

## References
* R. C. Martin, *Clean Code: A Handbook of Agile Software Craftsmanship*, &sect;17.G28. Prentice Hall, 2008.
* S. McConnell, *Code Complete: A Practical Handbook of Software Construction*. Microsoft Press, 2004.
