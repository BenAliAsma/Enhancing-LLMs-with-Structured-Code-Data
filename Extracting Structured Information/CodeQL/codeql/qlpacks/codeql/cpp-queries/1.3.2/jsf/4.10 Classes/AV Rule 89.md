# Inconsistent virtual inheritance
If a base class is both virtual and non-virtual within a single hierarchy then the hierarchy is more difficult to understand and maintain. This rule identifies any base classes that are derived as both virtual and non-virtual in one hierarchy as violations.


## References
* AV Rule 89, *Joint Strike Fighter Air Vehicle C++ Coding Standards*. Lockheed Martin Corporation, 2005.
* MSDN Library: [Multiple Base Classes](https://docs.microsoft.com/en-us/cpp/cpp/multiple-base-classes).
