# Superclass attribute shadows subclass method
Subclass shadowing occurs when an instance attribute of a superclass has the the same name as a method of a subclass, or vice-versa. The semantics of Python attribute look-up mean that the instance attribute of the superclass hides the method in the subclass.


## Recommendation
Rename the method in the subclass or rename the attribute in the superclass.


## Example
The following code includes an example of subclass shadowing. When you call `Cow().milk()` an error is raised because `Cow().milk` is interpreted as the 'milk' attribute set in `Mammal.__init__`, not the 'milk' method defined within `Cow`. This can be fixed by changing the name of either the 'milk' attribute or the 'milk' method.


```python
class Mammal(object):

    def __init__(self, milk = 0):
        self.milk = milk


class Cow(Mammal):

    def __init__(self):
        Mammal.__init__(self)

    def milk(self):
        return "Milk"

#Cow().milk() will raise an error as Cow().milk is the 'milk' attribute
#set in Mammal.__init__, not the 'milk' method defined on Cow.


```
