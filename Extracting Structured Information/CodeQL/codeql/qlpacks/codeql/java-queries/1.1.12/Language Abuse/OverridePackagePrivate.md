# Confusing non-overriding of package-private method
If a method is declared with default access (that is, not private, protected, nor public), it can only be overridden by methods in the same package. If a method of the same signature is defined in a subclass in a different package, it is a completely separate method and no overriding occurs.

Code like this can be confusing for other programmers, who have to understand that there is no overriding relation, check that the original programmer did not intend one method to override the other, and avoid mixing up the two methods by accident.


## Recommendation
In cases where there is intentionally no overriding, the best solution is to rename one or both of the methods to clarify their different purposes.

If one method is supposed to override another method that is declared with default access in another package, the access of the method must be changed to `public` or `protected`. Alternatively, the classes must be moved to the same package.


## Example
In the following example, `PhotoResizerWidget.width` does not override `Widget.width` because one method is in package `gui` and one method is in package `gui.extras`.


```java
// File 1
package gui;

abstract class Widget
{
    // ...

    // Return the width (in pixels) of this widget
    int width() {
        // ...
    }

    // ...
}

// File 2
package gui.extras;

class PhotoResizerWidget extends Widget
{
    // ...
 
    // Return the new width (of the photo when resized)
    public int width() {
        // ...
    }
   
    // ...
}
```
Assuming that no overriding is intentional, one or both of the methods should be renamed. For example, `PhotoResizerWidget.width` would be better named `PhotoResizerWidget.newPhotoWidth`.


## References
* Help - Eclipse Platform: [Java Compiler Errors/Warnings Preferences](https://help.eclipse.org/2020-12/advanced/content.jsp?topic=/org.eclipse.jdt.doc.user/reference/preferences/java/compiler/ref-preferences-errors-warnings.htm).
* Java Language Specification: [8.4.8.1 Overriding (by Instance Methods)](https://docs.oracle.com/javase/specs/jls/se11/html/jls-8.html#jls-8.4.8.1).
