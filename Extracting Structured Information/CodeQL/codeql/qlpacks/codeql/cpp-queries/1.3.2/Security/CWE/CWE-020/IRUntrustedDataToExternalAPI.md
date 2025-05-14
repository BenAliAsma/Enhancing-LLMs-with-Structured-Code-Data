# Untrusted data passed to external API
Using unsanitized untrusted data in an external API can cause a variety of security issues. This query reports external APIs that use untrusted data. The results are not filtered, so you can audit all examples. The query provides data for security reviews of the application and you can also use it to identify external APIs that should be modeled as either taint steps, or sinks for specific problems.

An external API is defined as a call to a function that is not defined in the source code, and is not modeled as a taint step in the default taint library. External APIs may be from the C++ standard library, third-party dependencies or from internal dependencies. The query reports uses of untrusted data in either the qualifier or as one of the arguments of external APIs.


## Recommendation
For each result:

* If the result highlights a known sink, confirm that the result is reported by the relevant query, or that the result is a false positive because this data is sanitized.
* If the result highlights an unknown sink for a problem, then add modeling for the sink to the relevant query, and confirm that the result is either found, or is safe due to appropriate sanitization.
* If the result represents a call to an external API that transfers taint, add the appropriate modeling, and re-run the query to determine what new results have appeared due to this additional modeling.
Otherwise, the result is likely uninteresting. Custom versions of this query can extend the `SafeExternalAPIFunction` class to exclude known safe external APIs from future analysis.


## Example
In this first example, input is read from `fgets` and then ultimately used in a call to the `fputs` external API:


```cpp
#include <cstdio>

void do_get(FILE* request, FILE* response) {
  char page[1024];
  fgets(page, 1024, request);

  char buffer[1024];
  strcat(buffer, "The page \"");
  strcat(buffer, page);
  strcat(buffer, "\" was not found.");

  fputs(buffer, response);
}
```
This is an XSS sink. The XSS query should therefore be reviewed to confirm that this sink is appropriately modeled, and if it is, to confirm that the query reports this particular result, or that the result is a false positive due to some existing sanitization.

In this second example, again a request parameter is read from `fgets`.


```cpp
#include <cstdio>

void do_get(FILE* request, FILE* response) {
  char user_id[1024];
  fgets(user_id, 1024, request);

  char buffer[1024];
  strcat(buffer, "SELECT * FROM user WHERE user_id='");
  strcat(buffer, user_id);
  strcat(buffer, "'");

  // ...
}
```
If the query reported the call to `strcat` on line 9, this would suggest that this external API is not currently modeled as a taint step in the taint tracking library. The next step would be to model this as a taint step, then re-run the query to determine what additional results might be found. In this example, it seems likely that `buffer` will be executed as an SQL query, potentially leading to an SQL injection vulnerability.

Note that both examples are correctly handled by the standard taint tracking library and XSS query.


## References
* Common Weakness Enumeration: [CWE-20](https://cwe.mitre.org/data/definitions/20.html).
