# Improper validation of code-specified array index
Using a hard-coded or randomly provided value as the index to an array access can cause that array access to throw an `ArrayIndexOutOfBoundsException` if the value is outside the bounds of that array.

This problem occurs when a literal value, or a value generated using the `Random`, is used as an index for an array access operation. If one or more of the range of values produced by the random operation, or the fixed value of the literal, is outside the bounds of the array then this can cause an `ArrayIndexOutOfBoundsException`.


## Recommendation
If the index is a literal value, then the literal value may need to be modified to specify an index that is guaranteed to lie within the bounds of the array. Alternatively, the literal value may represent a default value that was never intended to be used in the array access, in which case the logic should be modified to ensure the default value is never used.

For indices that flow from randomly generated values, either the random operation should be modified to generate a value that is guaranteed to be within the bounds of the array, or the array access should be protected by suitable conditional checks that verify the index is smaller than the length and larger than or equal to zero.


## Example
The following program searches through an array to find the index at which some search text matches:


```java
public class ImproperValidationOfArrayIndex extends HttpServlet {

  protected void doGet(HttpServletRequest request, HttpServletResponse response)
  throws ServletException, IOException {
    // Search for products in productDescriptions that match the search term
    String searchTerm = request.getParameter("productSearchTerm");
    int foundProductID = -1;
    for (int i = 0; i < productDescriptions.length; i++) {
      if (productDescriptions[i].contains(searchTerm)) {
        // Found matching product
        foundProductID = i;
        break;
      }
    }

    // BAD We may not have found a product in which case the index would be -1
    response.getWriter().write(productDescriptions[foundProductID]);

    if (foundProductID >= 0) {
      // GOOD We have checked we found a product first
      response.getWriter().write(productDescriptions[foundProductID]);
    } else {
      response.getWriter().write("No product found");
    }
  }
}
```
If the search text is not found, `foundProductID` is set to the default value - specified as `-1`. In the first access, `foundProductID` is used without checking whether the index is `-1`. This code can therefore throw a `ArrayIndexOutOfBoundsException` if the search text is not found.

In the second case, the array access is protected by a conditional that verifies the index is greater than or equal to zero. This ensures that an `ArrayIndexOutOfBoundsException` cannot be thrown.


## References
* Java API Specification: [ArrayIndexOutOfBoundsException](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/ArrayIndexOutOfBoundsException.html).
* Common Weakness Enumeration: [CWE-129](https://cwe.mitre.org/data/definitions/129.html).
