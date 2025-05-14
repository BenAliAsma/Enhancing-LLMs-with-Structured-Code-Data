# Equals on incomparable types
Calls of the form `x.equals(y)`, where `x` and `y` have incomparable types, should always return `false` because the runtime types of `x` and `y` will be different. Two types are incomparable if they are distinct and do not have a common subtype.


## Recommendation
Ensure that such comparisons use comparable types.


## Example
In the following example, the call to `equals` on line 5 refers to the whole array by mistake, instead of a specific element. Therefore, "Value not found" is returned.


```java
String[] anArray = new String[]{"a","b","c"}
String valueToFind = "b";

for(int i=0; i<anArray.length; i++){
  if(anArray.equals(valueToFind){    // anArray[i].equals(valueToFind) was intended
    return "Found value at index " + i;
  }
}

return "Value not found";
```

## References
* Java API Specification: [Object.equals()](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/Object.html#equals(java.lang.Object)).
