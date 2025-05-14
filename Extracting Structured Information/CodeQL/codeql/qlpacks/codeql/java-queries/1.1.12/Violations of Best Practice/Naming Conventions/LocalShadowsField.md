# Local variable shadows field
This query finds local variables that shadow like-named field declarations. This is confusing since it might easily lead to assignments to the local variable that should have been to the corresponding field.


## How to Address the Query Results
For clarity, it may be better to rename the variable to avoid shadowing.


## References
