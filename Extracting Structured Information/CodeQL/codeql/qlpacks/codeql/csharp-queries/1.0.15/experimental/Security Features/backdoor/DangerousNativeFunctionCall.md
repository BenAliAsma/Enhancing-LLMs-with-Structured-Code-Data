# Potential dangerous use of native functions
This query finds native calls to external functions that are often used in creating backdoors or are generally attributed to unsafe code practices. This is an example of a query that may be useful for detecting potential backdoors. Solorigate is one example that uses this mechanism.


## Recommendation
Any findings from this rule are only intended to indicate suspicious code that shares similarities with known portions of code used for the Solorigate attack. There is no certainty that the code is related or that the code is part of any attack.

For more information about Solorigate, please visit https://aka.ms/solorigate.

