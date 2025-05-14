# Inherited equals() in subclass with added fields
If a class overrides the default implementation of equality defined by the `Object.equals` method, and a subclass of that class declares additional fields to the ones that it inherits, the results of `equals` may be wrong, unless that subclass also redefines `equals`.


## Recommendation
See if the subclass should provide its own implementation of `equals` to take into account the additional fields that it declares.

If the subclass cannot provide its own implementation of `equals` because the inherited `equals` method is `final`, consider replacing inheritance by composition; instead of class `B` extending class `A`, class `B` could define a field of type `A`.


## Example
In the following example, rectangles `r1` and `r2` are calculated to be equal, even though they have different dimensions. This is because the class `Rectangle` does not override `Square.equals`, so it uses a test for equality that is only applicable to squares, not rectangles. (Note that, in practice, the example should also include an implementation of `hashCode`.)


```java
public class DefineEqualsWhenAddingFields {
    static class Square {
        protected int width = 0;
        public Square(int width) {
            this.width = width;
        }
        @Override
        public boolean equals(Object thatO) {  // This method works only for squares.
            if(thatO != null && getClass() == thatO.getClass() ) {
                Square that = (Square)thatO;
                return width == that.width;
            }
            return false;
        }
    }

    static class Rectangle extends Square {
        private int height = 0;
        public Rectangle(int width, int height) {
            super(width);
            this.height = height;
        }
    }

    public static void main(String[] args) {
        Rectangle r1 = new Rectangle(4, 3);
        Rectangle r2 = new Rectangle(4, 5);
        System.out.println(r1.equals(r2));  // Outputs 'true'
    }
}

```
To get the correct result, you must override `Square.equals` in class `Rectangle`.


## References
* Java API Specification: [Object.equals()](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/Object.html#equals(java.lang.Object)).
