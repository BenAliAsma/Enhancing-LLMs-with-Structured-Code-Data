# Missing Locale parameter to toUpperCase() or toLowerCase()
The parameterless versions of `String.toUpperCase()` and `String.toLowerCase()` use the default locale of the Java Virtual Machine when transforming strings. This can cause unexpected behavior for certain locales.


## Recommendation
Use the corresponding methods with explicit locale parameters to ensure that the results are consistent across all locales. For example:

`System.out.println("I".toLowerCase(java.util.Locale.ENGLISH));`

prints `i`, regardless of the default locale.


## Example
In the following example, the calls to the parameterless functions may return different strings for different locales. For example, if the default locale is ENGLISH, the function `toLowerCase()` converts a capital `I` to `i`; if the default locale is TURKISH, the function `toLowerCase()` converts a capital `I` to the Unicode Character "Latin small letter dotless i" (U+0131).

To ensure that an English string is returned, regardless of the default locale, the example shows how to call `toLowerCase` and pass `locale.ENGLISH` as the argument. (This assumes that the text is English. If the text is Turkish, you should pass `locale.TURKISH` as the argument.)


```java
public static void main(String args[]) {
    String phrase = "I miss my home in Mississippi.";

    // AVOID: Calling 'toLowerCase()' or 'toUpperCase()'
    // produces different results depending on what the default locale is.
    System.out.println(phrase.toUpperCase());
    System.out.println(phrase.toLowerCase());

    // GOOD: Explicitly setting the locale when calling 'toLowerCase()' or
    // 'toUpperCase()' ensures that the resulting string is
    // English, regardless of the default locale.
    System.out.println(phrase.toLowerCase(Locale.ENGLISH));
    System.out.println(phrase.toUpperCase(Locale.ENGLISH));
}
```

## References
* Java API Specification: [String.toUpperCase()](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/String.html#toUpperCase()).
