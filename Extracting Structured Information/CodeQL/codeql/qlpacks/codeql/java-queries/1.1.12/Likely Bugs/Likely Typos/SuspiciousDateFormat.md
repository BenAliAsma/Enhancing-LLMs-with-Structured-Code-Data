# Suspicious date format
The Java `SimpleDateFormat` class provides many placeholders so that you can define precisely the date format required. However, this also makes it easy to define a pattern that doesn't behave exactly as you intended. The most common mistake is to use the `Y` placeholder (which represents the ISO 8601 week year), rather than `y` (which represents the actual year). In this case, the date reported will appear correct until the end of the year, when the "week year" may differ from the actual year.


## Recommendation
Ensure the format pattern's use of `Y` is correct, and if not replace it with `y`.


## Example
The following example uses the date format `YYYY-MM-dd`. On the 30th of December 2019, this code will output "2020-12-30", rather than the intended "2019-12-30".


```java
System.out.println(new SimpleDateFormat("YYYY-MM-dd").format(new Date()));

```
The correct pattern in this case would be `yyyy-MM-dd` instead of `YYYY-MM-dd`.


## References
* Java API Specification: [SimpleDateFormat](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/text/SimpleDateFormat.html).
