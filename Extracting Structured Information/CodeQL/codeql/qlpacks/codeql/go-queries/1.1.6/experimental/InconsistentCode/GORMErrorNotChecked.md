# GORM error not checked
GORM errors are returned as a field of the return value instead of a separate return value.

It is therefore very easy to miss that an error may occur and omit error handling routines.


## Recommendation
Ensure that GORM errors are checked.


## Example
In the example below, the error from the database query is never checked:


```go
package main

import "gorm.io/gorm"

func getUserId(db *gorm.DB, name string) int64 {
	var user User
	db.Where("name = ?", name).First(&user)
	return user.Id
}

```
The corrected version checks and handles the error before returning.


```go
package main

import "gorm.io/gorm"

func getUserIdGood(db *gorm.DB, name string) int64 {
	var user User
	if err := db.Where("name = ?", name).First(&user).Error; err != nil {
		// handle errors
	}
	return user.Id
}

```

## References
* [GORM Error Handling](https://gorm.io/docs/error_handling.html).
